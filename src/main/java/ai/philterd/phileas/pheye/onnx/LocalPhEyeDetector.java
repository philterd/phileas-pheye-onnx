/*
 *     Copyright 2025 Philterd, LLC @ https://www.philterd.ai
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *          http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.philterd.phileas.pheye.onnx;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.philterd.phileas.services.filters.ai.pheye.PhEyeDetector;
import ai.philterd.phileas.services.filters.ai.pheye.PhEyeSpan;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * On-device GLiNER inference via ONNX Runtime. This is a Java port of the GLiNER 0.2.25
 * uni-encoder span pipeline (the recipe PhEye uses): build an entity prompt, tokenize prompt+text,
 * enumerate candidate word spans, run the ONNX model, then sigmoid + threshold + greedy
 * (flat, non-overlapping) decode, mapping word spans back to character offsets.
 *
 * <p><b>Model directory layout</b> (from {@code ./hub/publish.sh}):
 * <ul>
 *   <li>{@code model.onnx} (or {@code onnx/model.onnx}) — the exported GLiNER model</li>
 *   <li>{@code tokenizer.json} — the HuggingFace fast tokenizer</li>
 *   <li>{@code gliner_config.json} — span width, max length, prompt tokens</li>
 * </ul>
 *
 * <p><b>Parity gate (important):</b> a redaction model that decodes spans incorrectly leaks
 * names. The exact ONNX tensor dtypes/shapes and the {@code logits} layout depend on the
 * gliner exporter, and tokenization parity depends on the DJL tokenizer version. This class
 * must be validated against an exported model with {@code LocalPhEyeDetectorParityTest}
 * (compare to Python {@code gliner.predict_entities}) before it is trusted in production.
 */
public class LocalPhEyeDetector implements PhEyeDetector {

    // Default confidence threshold; the PhEyeFilter applies its own per-label thresholds on top.
    private static final double DEFAULT_THRESHOLD = 0.5;

    private final GlinerConfig config;
    private final HuggingFaceTokenizer tokenizer;
    private final OrtEnvironment ortEnvironment;
    private final OrtSession session;

    public LocalPhEyeDetector(final Path modelDir) throws Exception {

        this.config = GlinerConfig.load(modelDir);

        if (!"whitespace".equals(config.wordsSplitterType)) {
            throw new IllegalArgumentException("Unsupported words_splitter_type '" + config.wordsSplitterType
                    + "'. Only 'whitespace' is supported by this detector.");
        }

        this.tokenizer = HuggingFaceTokenizer.newInstance(modelDir.resolve("tokenizer.json"));

        final Path onnx = resolveOnnxPath(modelDir);
        this.ortEnvironment = OrtEnvironment.getEnvironment();
        this.session = ortEnvironment.createSession(onnx.toString(), new OrtSession.SessionOptions());

    }

    private static Path resolveOnnxPath(final Path modelDir) {
        final Path nested = modelDir.resolve("onnx").resolve("model.onnx");
        if (Files.exists(nested)) {
            return nested;
        }
        return modelDir.resolve("model.onnx");
    }

    @Override
    public List<PhEyeSpan> detect(final String text, final Collection<String> labels,
                                  final String context, final int piece) throws Exception {

        final List<String> labelList = new ArrayList<>(labels);
        if (labelList.isEmpty() || text == null || text.isBlank()) {
            return new ArrayList<>();
        }

        // 1. Split into words with char offsets, then truncate to the model's max length.
        List<WordsSplitter.Word> words = WordsSplitter.split(text);
        if (words.size() > config.maxLen) {
            words = words.subList(0, config.maxLen);
        }
        final int numWords = words.size();
        if (numWords == 0) {
            return new ArrayList<>();
        }

        // 2. Build the prompt word list: [<<ENT>> label]* <<SEP>> <text words...>
        //    (mirrors SpanProcessor.prepare_inputs; markerV0 prompt).
        final List<String> inputWords = new ArrayList<>();
        for (final String label : labelList) {
            inputWords.add(config.entToken);
            inputWords.add(label);
        }
        inputWords.add(config.sepToken);
        final int promptWordCount = inputWords.size();
        for (final WordsSplitter.Word w : words) {
            inputWords.add(w.text());
        }

        // 3. Tokenize pre-split (is_split_into_words=True). The tokenizer adds special tokens.
        final Encoding encoding = tokenizer.encode(inputWords.toArray(new String[0]));
        final long[] inputIds = encoding.getIds();
        final long[] attentionMask = encoding.getAttentionMask();
        final long[] wordIds = encoding.getWordIds(); // word index per token; -1 for special tokens
        final int seqLen = inputIds.length;

        // words_mask: first subtoken of each TEXT word gets its 1-based text-word index, else 0
        // (mirrors prepare_word_mask with skip_first_words=promptWordCount).
        final long[] wordsMask = new long[seqLen];
        long previousWordId = Long.MIN_VALUE;
        for (int t = 0; t < seqLen; t++) {
            final long wid = wordIds[t];
            if (wid >= promptWordCount && wid != previousWordId) {
                wordsMask[t] = wid - promptWordCount + 1;
            } else {
                wordsMask[t] = 0;
            }
            previousWordId = wid;
        }

        // 4. Enumerate candidate spans [i, i+width] for width in 0..maxWidth-1; mask out-of-range.
        final int spansPerWord = config.maxWidth;
        final int numSpans = numWords * spansPerWord;
        final long[][] spanIdx = new long[numSpans][2];
        final long[] spanMask = new long[numSpans];
        int s = 0;
        for (int i = 0; i < numWords; i++) {
            for (int k = 0; k < spansPerWord; k++) {
                final int end = i + k;
                spanIdx[s][0] = i;
                spanIdx[s][1] = end;
                spanMask[s] = (end < numWords) ? 1L : 0L;
                s++;
            }
        }

        // 5. Run the ONNX model.
        final float[][][] logits = runModel(inputIds, attentionMask, wordsMask, numWords, spanIdx, spanMask, numWords);

        // 6. Decode: sigmoid + threshold -> candidate spans, then greedy non-overlap by score.
        final List<Candidate> candidates = new ArrayList<>();
        for (int i = 0; i < numWords; i++) {
            for (int k = 0; k < spansPerWord; k++) {
                final int end = i + k;
                if (end >= numWords) {
                    continue;
                }
                for (int c = 0; c < labelList.size(); c++) {
                    final double prob = sigmoid(logits[i][k][c]);
                    if (prob > DEFAULT_THRESHOLD) {
                        candidates.add(new Candidate(i, end, c, prob));
                    }
                }
            }
        }

        final List<Candidate> selected = greedyNonOverlap(candidates);

        // 7. Map word spans -> char offsets -> PhEyeSpan.
        final List<PhEyeSpan> spans = new ArrayList<>();
        for (final Candidate cand : selected) {
            final int startChar = words.get(cand.startWord).start();
            final int endChar = words.get(cand.endWord).end();

            final PhEyeSpan span = new PhEyeSpan();
            span.setStart(startChar);
            span.setEnd(endChar);
            span.setLabel(labelList.get(cand.classIndex));
            span.setText(text.substring(startChar, endChar));
            span.setScore(cand.score);
            spans.add(span);
        }

        return spans;

    }

    /**
     * Run the span model and return logits shaped [numWords][maxWidth][numClasses].
     *
     * <p>VERIFY against the exported model: the input tensor dtypes (int64 here) and the
     * raw {@code logits} output shape must match the gliner export. GLiNER's span model emits
     * logits of [batch, numWords, maxWidth, numClasses]; this method reshapes to that.
     */
    private float[][][] runModel(final long[] inputIds, final long[] attentionMask, final long[] wordsMask,
                                 final int textLength, final long[][] spanIdx, final long[] spanMask,
                                 final int numWords) throws Exception {

        final int seqLen = inputIds.length;
        final int numSpans = spanIdx.length;
        final int numClasses; // inferred from the output below

        final Map<String, OnnxTensor> inputs = new HashMap<>();
        try {

            inputs.put("input_ids", OnnxTensor.createTensor(ortEnvironment, new long[][]{inputIds}));
            inputs.put("attention_mask", OnnxTensor.createTensor(ortEnvironment, new long[][]{attentionMask}));
            inputs.put("words_mask", OnnxTensor.createTensor(ortEnvironment, new long[][]{wordsMask}));
            inputs.put("text_lengths", OnnxTensor.createTensor(ortEnvironment, new long[][]{{textLength}}));
            inputs.put("span_idx", OnnxTensor.createTensor(ortEnvironment, new long[][][]{spanIdx}));
            inputs.put("span_mask", OnnxTensor.createTensor(ortEnvironment, new long[][]{spanMask}));

            try (final OrtSession.Result result = session.run(inputs)) {

                final OnnxValue value = result.get("logits").orElseThrow(
                        () -> new IllegalStateException("ONNX model did not return a 'logits' output."));
                final float[] flat = flatten(((OnnxTensor) value).getFloatBuffer().array());

                // Reshape flat logits to [numWords][maxWidth][numClasses].
                numClasses = flat.length / (numWords * config.maxWidth);
                final float[][][] logits = new float[numWords][config.maxWidth][numClasses];
                int idx = 0;
                for (int i = 0; i < numWords; i++) {
                    for (int k = 0; k < config.maxWidth; k++) {
                        for (int c = 0; c < numClasses; c++) {
                            logits[i][k][c] = flat[idx++];
                        }
                    }
                }
                return logits;

            }

        } finally {
            for (final OnnxTensor tensor : inputs.values()) {
                tensor.close();
            }
        }

    }

    private static float[] flatten(final float[] array) {
        return array;
    }

    /** Greedy flat NER: keep highest-scoring spans that do not overlap already-kept ones. */
    private static List<Candidate> greedyNonOverlap(final List<Candidate> candidates) {
        candidates.sort((a, b) -> Double.compare(b.score, a.score));
        final List<Candidate> kept = new ArrayList<>();
        for (final Candidate c : candidates) {
            boolean overlaps = false;
            for (final Candidate k : kept) {
                if (c.startWord <= k.endWord && k.startWord <= c.endWord) {
                    overlaps = true;
                    break;
                }
            }
            if (!overlaps) {
                kept.add(c);
            }
        }
        kept.sort((a, b) -> Integer.compare(a.startWord, b.startWord));
        return kept;
    }

    private static double sigmoid(final double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    @Override
    public void close() throws Exception {
        if (session != null) {
            session.close();
        }
        if (tokenizer != null) {
            tokenizer.close();
        }
    }

    /** A scored candidate span over word indices (inclusive) for a given class. */
    private record Candidate(int startWord, int endWord, int classIndex, double score) {}

}
