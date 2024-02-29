apo_meta_prompt = """{{#system~}}
You are a helpful assistant.
{{/system ̃}}

{{#user ̃}}
I'm trying to write a zero-shot classifier prompt.

My current prompt is:
"{{prompt}}"


But this prompt gets the following examples wrong:
{{failure_string}}

Give {{n_reasons}} reasons why the prompt could have gotten these examples wrong. Do not include other text.
{{/user ̃}}

{{#assistant ̃}}
{{gen 'gradients' temperature=0.0}}
{{/assistant ̃}}"""

apo_refine_meta_prompt = """{{#system ̃}}
You are a helpful assistant.
{{/system ̃}}

{{#user ̃}}
I'm trying to write a zero-shot classifier.

My current prompt is:
"{{prompt}}"

But it gets the following examples wrong:
{{failure_string}}

Based on these examples the problem with this prompt is that:
{{gradient}}

Based on the above information, I wrote an improved prompt. The total length of the prompt should be less than {{max_tokens}} words. Please output only prompt and nothing else. Please modify it slightly, not too much.
{{/user ̃}}

{{#assistant ̃}}
{{gen 'new_prompt' temperature=0.0}}
{{/assistant ̃}}
"""
