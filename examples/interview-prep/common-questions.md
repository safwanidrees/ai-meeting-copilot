# Common interview questions: my prepared answers

## Tell me about yourself

I am an AI engineer who enjoys building AI systems that work in real time. My recent project is an AI Meeting Copilot: it listens to meeting audio, transcribes it locally with Whisper, searches my own documents with hybrid retrieval, and streams a suggested answer in about two seconds. Building it taught me real-time audio processing, retrieval augmented generation, working with several LLM providers, and writing tests that catch real bugs.
EDIT ME: add your education or experience in one sentence.

## What is your greatest strength?

I verify instead of assuming. On my meeting copilot every test passed on the first run, which made me suspicious, so I deliberately broke the code to prove the tests could fail, and that process uncovered a flaky test and a real concurrency bug.

## What is a weakness you are working on?

I tend to go deep into technical details, which can slow down early decisions. I now time-box exploration, ship a simple working version first, and improve it based on measurements.

## Describe a difficult bug you solved

My retrieval pipeline returned no keyword matches for small document sets. I traced it to the Okapi BM25 IDF formula, which becomes negative when a term appears in more than half of the documents, so exact matches scored below zero and were filtered out. I implemented BM25 with Lucene's always-positive IDF and added a regression test with a one-document corpus.

## Why do you want this role?

EDIT ME: connect the company's product or mission to something you have built or want to learn.

## Do you have any questions for us?

What does success look like in the first three months in this role? How does the team evaluate and monitor its machine learning models in production? What is the biggest technical challenge the team is facing right now?
