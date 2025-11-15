import os
import groq
import asyncio
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import nltk
import json
import logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("adrsa_framework")

class Polarity(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

@dataclass
class AspectSentiment:
    aspect: str
    polarity: Polarity
    confidence: float

class ADRSAAgent:
    def __init__(self, groq_api_key: str):
        logger.info("Initializing ADRSAAgent")
        if not groq_api_key:
            logger.warning("No GROQ_API_KEY provided; Groq client may fail at runtime")

        self.client = groq.Groq(api_key=groq_api_key)
        self.aspect_vocab = set()
        self.sentiment_vocab = set()
        self.negation_words = {'not', 'no', 'never', 'nothing', 'none', 'neither', 'nor'}
        logger.debug("Agent vocabularies initialized: negation_words=%s", self.negation_words)
        
    async def process_review(self, review_text: str) -> List[AspectSentiment]:
        """Main pipeline for processing review text"""
        logger.info("Processing review: %d characters", len(review_text or ""))
        sentences = await self.preprocessing_agent(review_text)

        logger.info("Preprocessing produced %d sentences", len(sentences))
        results = []
        for idx, sentence in enumerate(sentences, start=1):
            logger.info("Processing sentence %d/%d: %s", idx, len(sentences), sentence)

            try:
                aspect_agent_result = await self.aspect_detection_agent(sentence)
                logger.debug("Aspects detected: %s", aspect_agent_result.get('aspects'))

                sentiment_agent_result = await self.sentiment_detection_agent(sentence)
                logger.debug("Sentiment words detected: %s", sentiment_agent_result.get('sentiment_words'))

                negation_agent_result = await self.negation_detection_agent(sentence)
                logger.debug("Negation detection result: %s", negation_agent_result)

                label_pairs = await self.label_pair_generation_agent(
                    aspect_agent_result,
                    sentiment_agent_result,
                    negation_agent_result,
                    sentence,
                )

                logger.info("Generated %d label pairs for sentence %d", len(label_pairs), idx)
                results.extend(label_pairs)
            except Exception:
                logger.exception("Unexpected error while processing sentence %d; skipping to next", idx)
                continue

        logger.info("Processing complete. Total aspect-polarity pairs: %d", len(results))
        return results
    
    async def preprocessing_agent(self, text: str) -> List[str]:
        """Agent 1: Text preprocessing and sentence splitting"""
        prompt = f"""
        Split the following review text into individual sentences. Remove any sentences that don't contain sentiment words or are irrelevant for analysis.
        
        Review Text: {text}
        
        Return only the meaningful sentences as a JSON list.
        """
        
        logger.debug("Calling preprocessing_agent LLM with model=%s", "llama-3.3-70b-versatile")
        raw = None
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            raw = getattr(response.choices[0].message, "content", None)
        except Exception:
            logger.exception("LLM call failed in preprocessing_agent")

        logger.debug("Preprocessing response length: %d", len(raw or ""))
        sentences = self._parse_sentences_response(raw)

        filtered_sentences = [s for s in sentences if self._has_sentiment_words(s)]
        logger.info("Filtered to %d meaningful sentences after sentiment-word check", len(filtered_sentences))

        return filtered_sentences
    
    async def aspect_detection_agent(self, sentence: str) -> Dict:
        """Agent 2: Aspect detection using n-gram models"""
        prompt = f"""
        Analyze the following sentence and identify all product/service aspects mentioned. 
        Use n-gram analysis to detect multi-word aspects.
        
        Sentence: "{sentence}"
        
        Return aspects as a JSON with:
        - "aspects": list of detected aspects
        - "ngrams": relevant n-grams found
        - "confidence": confidence score for each aspect
        """
        
        logger.debug("Calling aspect_detection_agent for sentence: %s", sentence)
        raw = None
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw = getattr(response.choices[0].message, "content", None)
        except Exception:
            logger.exception("LLM call failed in aspect_detection_agent")

        logger.debug("Aspect detection response length: %d", len(raw or ""))
        return self._parse_aspect_response(raw, sentence)
    
    async def sentiment_detection_agent(self, sentence: str) -> Dict:
        """Agent 3: Sentiment word detection"""
        prompt = f"""
        Identify all sentiment-bearing words and phrases in the following sentence. 
        Classify each sentiment word as positive, negative, or neutral.
        
        Sentence: "{sentence}"
        
        Return as JSON with:
        - "sentiment_words": list of detected sentiment words/phrases
        - "polarities": polarity for each word
        - "intensities": intensity score (0-1)
        """
        
        logger.debug("Calling sentiment_detection_agent for sentence: %s", sentence)
        raw = None
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            raw = getattr(response.choices[0].message, "content", None)
        except Exception:
            logger.exception("LLM call failed in sentiment_detection_agent")

        logger.debug("Sentiment detection response length: %d", len(raw or ""))
        return self._parse_sentiment_response(raw)
    
    async def negation_detection_agent(self, sentence: str) -> Dict:
        """Agent 4: Negation detection and scope identification"""
        prompt = f"""
        Detect negation patterns in the following sentence and identify their scope.
        
        Sentence: "{sentence}"
        
        Return as JSON with:
        - "has_negation": boolean
        - "negation_words": list of negation terms found
        - "negation_scope": the phrases affected by negation
        - "flipped_polarity": whether polarity should be flipped
        """
        
        logger.debug("Calling negation_detection_agent for sentence: %s", sentence)
        raw = None
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            raw = getattr(response.choices[0].message, "content", None)
        except Exception:
            logger.exception("LLM call failed in negation_detection_agent")

        logger.debug("Negation detection response length: %d", len(raw or ""))
        return self._parse_negation_response(raw)
    
    async def label_pair_generation_agent(self, aspect_data: Dict, sentiment_data: Dict, 
                                        negation_data: Dict, sentence: str) -> List[AspectSentiment]:
        """Agent 5: Generate final (aspect, polarity) pairs"""
        prompt = f"""
        Based on the analysis, generate aspect-polarity pairs for the sentence.
        
        Sentence: "{sentence}"
        Aspects: {aspect_data.get('aspects', [])}
        Sentiment Words: {sentiment_data.get('sentiment_words', [])}
        Negation: {negation_data.get('has_negation', False)}
        
        For each aspect, determine the most appropriate sentiment polarity considering:
        - Proximity to sentiment words
        - Negation effects
        - Contextual meaning
        
        Return as JSON list of {{"aspect": "...", "polarity": "positive/negative/neutral", "confidence": 0.0-1.0}}
        """
        
        logger.debug("Calling label_pair_generation_agent for sentence: %s", sentence)
        raw = None
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            raw = getattr(response.choices[0].message, "content", None)
        except Exception:
            logger.exception("LLM call failed in label_pair_generation_agent")

        logger.debug("Label pair generation response length: %d", len(raw or ""))
        return self._parse_label_pairs(raw)
    
    def _parse_sentences_response(self, response: str) -> List[str]:
        """Parse sentences from LLM response"""
        if not response:
            logger.debug("Empty preprocessing response; returning empty sentence list")
            return []

        json_block = self._extract_json_from_text(response)
        if json_block:
            try:
                parsed = json.loads(json_block)
                if isinstance(parsed, list):
                    return [s.strip() for s in parsed if isinstance(s, str) and s.strip()]
            except Exception:
                logger.debug("Extracted JSON block failed to parse; falling back to regex/split")

        logger.debug("Falling back to regex sentence split for preprocessing response")
        return [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
    
    def _parse_aspect_response(self, response: str, sentence: str) -> Dict:
        """Parse aspect detection response"""
        if not response:
            logger.debug("No aspect response from LLM; using POS-tag fallback")
            words = nltk.word_tokenize(sentence)
            pos_tags = nltk.pos_tag(words)
            aspects = [word for word, pos in pos_tags if pos.startswith('NN')]
            logger.info("Fallback extracted %d aspects using POS tags", len(aspects))
            return {'aspects': aspects, 'confidence': 0.7}

        json_block = self._extract_json_from_text(response)
        if json_block:
            try:
                data = json.loads(json_block)
                for aspect in data.get('aspects', []):
                    try:
                        self.aspect_vocab.add(aspect.lower())
                    except Exception:
                        logger.debug("Non-string aspect skipped: %s", repr(aspect))

                logger.info("Parsed %d aspects from LLM", len(data.get('aspects', [])))
                return data
            except Exception:
                logger.exception("Failed to parse extracted JSON for aspects; falling back to POS-tag")

        logger.debug("Could not extract JSON from aspect response; using POS-tag fallback")
        words = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(words)
        aspects = [word for word, pos in pos_tags if pos.startswith('NN')]
        logger.info("Fallback extracted %d aspects using POS tags", len(aspects))
        return {'aspects': aspects, 'confidence': 0.7}
    
    def _parse_sentiment_response(self, response: str) -> Dict:
        """Parse sentiment detection response"""
        if not response:
            logger.debug("No sentiment response from LLM; returning empty sentiment data")
            return {'sentiment_words': [], 'polarities': [], 'intensities': []}

        json_block = self._extract_json_from_text(response)
        if json_block:
            try:
                data = json.loads(json_block)
                for word in data.get('sentiment_words', []):
                    try:
                        self.sentiment_vocab.add(word.lower())
                    except Exception:
                        logger.debug("Non-string sentiment word skipped: %s", repr(word))

                logger.info("Parsed %d sentiment words from LLM", len(data.get('sentiment_words', [])))
                return data
            except Exception:
                logger.exception("Failed to parse extracted JSON for sentiment; returning empty sentiment data")

        logger.debug("Could not extract JSON from sentiment response; returning empty sentiment data")
        return {'sentiment_words': [], 'polarities': [], 'intensities': []}
    
    def _parse_negation_response(self, response: str) -> Dict:
        """Parse negation detection response"""
        if not response:
            logger.debug("No negation response from LLM; assuming no negation")
            return {'has_negation': False, 'negation_words': [], 'flipped_polarity': False}

        json_block = self._extract_json_from_text(response)
        if json_block:
            try:
                return json.loads(json_block)
            except Exception:
                logger.exception("Failed to parse extracted JSON for negation; assuming no negation")

        logger.debug("Could not extract JSON from negation response; assuming no negation")
        return {'has_negation': False, 'negation_words': [], 'flipped_polarity': False}
    
    def _parse_label_pairs(self, response: str) -> List[AspectSentiment]:
        """Parse final label pairs"""
        if not response:
            logger.debug("No label pair response from LLM; returning empty list")
            return []

        json_block = self._extract_json_from_text(response)
        if not json_block:
            logger.debug("Could not extract JSON from label pair response; returning empty list")
            return []

        try:
            pairs_data = json.loads(json_block)
            results = []
            if not isinstance(pairs_data, list):
                logger.debug("Label pair JSON was not a list; attempting to coerce")
                pairs_data = [pairs_data]

            for pair in pairs_data:
                try:
                    aspect = pair.get('aspect') if isinstance(pair, dict) else None
                    polarity_raw = None
                    if isinstance(pair, dict):
                        polarity_raw = pair.get('polarity')

                    polarity_value = None
                    if isinstance(polarity_raw, str):
                        normalized = polarity_raw.strip().lower()
                        if normalized in ('positive', 'pos', 'p'):
                            polarity_value = Polarity.POSITIVE
                        elif normalized in ('negative', 'neg', 'n'):
                            polarity_value = Polarity.NEGATIVE
                        else:
                            polarity_value = Polarity.NEUTRAL
                    else:
                        polarity_value = Polarity.NEUTRAL

                    confidence = 0.5
                    try:
                        confidence = float(pair.get('confidence', 0.5))
                    except Exception:
                        logger.debug("Invalid confidence value in pair; defaulting to 0.5: %s", repr(pair.get('confidence', None)))

                    if not aspect:
                        logger.debug("Skipping label pair with missing aspect: %s", repr(pair))
                        continue

                    results.append(AspectSentiment(
                        aspect=str(aspect),
                        polarity=polarity_value,
                        confidence=max(0.0, min(1.0, confidence)),
                    ))
                except Exception:
                    logger.exception("Skipping invalid label pair entry: %s", repr(pair))

            logger.info("Parsed %d aspect-polarity pairs from LLM", len(results))
            return results
        except Exception:
            logger.exception("Failed to parse label pairs; returning empty list")
            return []

    def _extract_json_from_text(self, text: str) -> str:
        """Attempt to extract a JSON object/array from noisy LLM text.

        Strategy:
        - Try parsing the whole text first.
        - If that fails, scan for the first '{' or '[' and find a matching closing brace
          while respecting string quoting and escapes. Return the first valid JSON
          substring that parses.
        - Return None if nothing valid found.
        """
        if not text:
            return None

        try:
            json.loads(text)
            return text
        except Exception:
            pass

        for start_char in ('{', '['):
            start = text.find(start_char)
            if start == -1:
                continue

            stack = []
            in_str = False
            escape = False
            for i in range(start, len(text)):
                c = text[i]
                if c == '"' and not escape:
                    in_str = not in_str
                if in_str:
                    if c == '\\' and not escape:
                        escape = True
                    else:
                        escape = False
                    continue

                if c == start_char:
                    stack.append(c)
                elif start_char == '{' and c == '}':
                    if stack:
                        stack.pop()
                    if not stack:
                        candidate = text[start:i+1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except Exception:
                            break
                elif start_char == '[' and c == ']':
                    if stack:
                        stack.pop()
                    if not stack:
                        candidate = text[start:i+1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except Exception:
                            break

        return None
    
    def _has_sentiment_words(self, sentence: str) -> bool:
        """Check if sentence contains sentiment words"""
        words = set(word.lower() for word in nltk.word_tokenize(sentence))
        has = bool(words & self.sentiment_vocab) or len(self.sentiment_vocab) == 0
        logger.debug("_has_sentiment_words: sentence='%s' has=%s (vocab_size=%d)", sentence, has, len(self.sentiment_vocab))
        return has

async def main():
    framework = ADRSAAgent(os.getenv("GROQ_API_KEY"))

    logger.info("=== Product Review Sentiment Analysis ===")
    print("=== Product Review Sentiment Analysis ===")
    print("Please enter your product review (press Enter twice to finish):")

    lines = []
    while True:
        line = input()
        if line == "" and lines:
            break
        if line:
            lines.append(line)

    review = " ".join(lines)

    if not review.strip():
        logger.info("No review provided by user; exiting")
        print("No review provided. Exiting.")
        return

    logger.info("User provided review (length=%d). Starting analysis.", len(review))
    print("\nAnalyzing your review...")

    results = await framework.process_review(review)

    logger.info("Analysis finished. Results count=%d", len(results))
    print("\nAspect-Sentiment Analysis Results:")
    print("=" * 40)
    for result in results:
        print(f"Aspect: {result.aspect:15} | Polarity: {result.polarity.value:8} | Confidence: {result.confidence:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
