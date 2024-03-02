# Prompts used with LLMs
`{text}` is classification sample text and `{target}` is a stance target.

## System command
System command used with gpt-3.5-turbo:

	You are a stance classifier

## Prompt *P3*
	From reading the tweet below, which of the options is most likely to be true about the tweeter's stance or outlook (explicit or implied) towards the target: {target}. Is it in favor, against or none?
	Tweet: {text}
	Answer with only one of 3 words: favor, against, none

## Prompt *P4*
	What position (explicit or implied) does the paragraph bellow take on the topic of: {target}? 
	{text}
	Answer with one of 3 words: favor, against or neither}


## Prompt *P5*
	An arbitrary user of social media webpage has posted the following:
	{text}
	You know nothing else about this user, but you should make assumptions regarding what he has posted. Make a best guess about an outlook of this user towards: {target}. Answer none if you aren't sure.
	Answer with only one of 3 words: favor, against or none



