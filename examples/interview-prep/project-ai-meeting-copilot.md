# Project: AI Meeting Copilot

## Tell me about this project (short version)

I built an open-source Python tool that listens to meeting audio, transcribes it in real time with Whisper, and streams a suggested answer and the likely follow-up question to the terminal. It can answer from your own documents using a hybrid RAG pipeline. Transcription takes roughly 0.4 to 1.3 seconds per utterance on a MacBook, and the project has 61 automated tests.

## Problem

In interviews, vivas and client calls you often need a precise answer quickly, and the relevant facts are spread across your notes and documents. I wanted a tool that hears the question, finds the right facts in my own material, and suggests an answer within a couple of seconds.

## Architecture

System audio is routed through a virtual device called BlackHole. The audio is converted to 16 kHz mono with a streaming resampler and a 100 Hz high-pass filter, then cut into 32 millisecond frames. Silero VAD scores each frame, and a small state machine decides when someone starts and stops speaking. Each finished utterance is transcribed locally with faster-whisper. The question is used to search my documents, and the best excerpts are sent with the question to an LLM (GPT-4o, Claude, or any model through OpenRouter), which streams the answer back token by token.

The program uses three threads connected by a queue: an audio thread, a keyboard input thread, and a worker thread that transcribes, retrieves and answers one job at a time. If the speaker keeps talking while an answer is streaming, the queued utterances are merged into one job instead of producing a burst of answers.

## Hybrid retrieval (RAG)

Documents are split into chunks of about 800 characters with 150 characters of overlap. Every question searches two indexes: BM25 for exact keywords, and vector embeddings stored in Qdrant for meaning. The two ranked lists are merged with Reciprocal Rank Fusion, which combines positions instead of raw scores because BM25 scores and cosine similarities are on different scales. Optionally a Cohere cross-encoder reranks the top 20 candidates, and the best 4 chunks go to the LLM. If the embedding or rerank API fails mid-meeting, retrieval falls back to whatever still works instead of crashing.

## Challenges and what I learned

The hardest bug: BM25 returned nothing for small document sets. The classic Okapi IDF formula becomes negative when a term appears in more than half of the documents, so with a single resume every exact match scored below zero and was filtered out. I replaced the library with my own BM25 using Lucene's always-positive IDF and added a regression test.

A concurrency bug: the audio thread printed status messages, but the display lock is held while an answer streams, so audio capture could block and eventually drop audio. I moved all printing off the audio thread and added a test that proves the audio thread never waits on the display.

Whisper hallucinates phrases such as "Thank you for watching" from silence, so I filter segments using Whisper's own no-speech probability, average log probability and compression ratio, plus a list of known hallucination phrases.

Free LLM models are often rate-limited, so I added automatic fallback models through OpenRouter, readable error messages, and a retry command.

## Results

It splits a noisy recording into the correct utterances, transcribes each one in about a second, and starts streaming an answer shortly after. It has 61 tests, including tests that I deliberately broke the code against to prove they catch real bugs.

## What I would improve next

Automatic meeting summaries and action items, rolling memory so the model remembers the whole meeting rather than the last six exchanges, autosaving the session after every exchange, and speaker labels.
