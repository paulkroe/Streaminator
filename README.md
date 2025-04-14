# MultiAnswerSpeculativeDecoding
Exploring speculative decoding for sampling multiple answer for a given question.

## Profiling:
1) We want a reasonable baseline to compare this to.
2) We want to understand the role of every single optimization. i.e. we want commandline args that determine which of the optimizations is turned on (one for kv cache optimization and one for continuous batching)
3) We want to stress test this (i.e. pass gsm8k through the whole thing)
4) We want to profile the GPU to understand whats happening

## optimizations(ideas):
1) Implement a scheduler that selects the next batch of prompts to be processed 
2) right now we copy the kv cache for the some prompts, there should be a way to not do that to reduce memory
3) there is a lot of sequential stuff in there. for example the prompt sampler, that should be made more efficient/paralell

## implementation stuff
support more models->kv cache wrapper


## main ideas:
implement ngrams to see if they work

## first i want to make sure the pipeline works. that means we should have a pass@8 score around 75% on gsm8k with llama 3.2 1B 

## metrics:
tokens per second