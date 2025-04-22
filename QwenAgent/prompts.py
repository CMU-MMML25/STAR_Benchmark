system_prompt = "You are a helpful video reasoning assistant."
user_prompt = """
You are a video reasoning assistant. When answering questions about video content:

- Use numbered steps to reason through the problem.
- When you need visual information, write a single line with: <look> your query </look>
  (This will retrieve frames from the video for you to use in your next step.)
- End with: Answer: <choice number>. <choice text>

Follow the examples below:

--------------------------------------------------
Example 1
Question: What does the chef do *after* adding salt?
Choices:
1. Tastes the soup.
2. Adds pepper.
3. Turns off the stove.
4. Puts on the lid.

Step 1: I need to see what the chef does right after adding salt.
<look> chef’s hands immediately after salt is poured </look>
Step 2: The chef grabs a pepper shaker and adds pepper to the pot.
Step 3: So the action that follows adding salt is adding pepper.
Answer: 2. Adds pepper.
--------------------------------------------------
Example 2
Question: Why does the girl scream?
Choices:
1. She sees a mouse.
2. She drops her ice cream.
3. Someone surprises her.
4. She wins a prize.

Step 1: First, check the scene right before the girl screams.
<look> frames right before the girl screams </look>
Step 2: A person jumps out from behind the door.
Step 3: So the girl screams because she was startled.
Answer: 3. Someone surprises her.
--------------------------------------------------

Now answer the following:

Question: {question}
Choices:
{choices}
"""


harder_user_prompt = """
You are a video reasoning assistant. When answering questions about video content:

- Use numbered steps to reason through the problem.
- When you need visual information, write a single line with: <look> your query </look>
  (This will retrieve frames from the video for you to use in your next step.)
- End with: Answer: <choice number>. <choice text>

Follow the examples below:

--------------------------------------------------
Example 1
Question: What causes the woman to run out of the house?
Choices:
1. She receives a phone call.
2. She hears a loud noise.
3. She sees something on TV.
4. She smells smoke in the kitchen.

Step 1: Start by checking what the woman is doing before she runs out.
<look> actions of the woman before she exits the house </look>
<assume there are some retrieved frames here>
Step 2: She is cooking in the kitchen and suddenly reacts to something.
Step 3: Check if there is something on the stove or in the kitchen that causes alarm.
<look> stove area and surroundings during cooking </look>
<assume there are some retrieved frames here>
Step 4: There is smoke rising from the pan.
Step 5: Check if she responds to the smoke or something else.
<look> her facial reaction and response after seeing smoke </look>
<assume there are some retrieved frames here>
Step 6: She covers her nose, looks panicked, and runs out.
Answer: 4. She smells smoke in the kitchen.
--------------------------------------------------
Example 2
Question: What did the boy do after realizing his toy was broken?
Choices:
1. He asked an adult for help.
2. He cried.
3. He threw it away.
4. He tried to fix it with tape.

Step 1: First, verify when the boy discovers the toy is broken.
<look> moment the boy inspects the toy </look>
<assume there are some retrieved frames here>
Step 2: He twists the toy, and a piece falls off.
Step 3: Check his immediate emotional reaction.
<look> his facial expression right after the toy breaks </look>
<assume there are some retrieved frames here>
Step 4: He looks upset but focused rather than sad.
Step 5: Check what he does next—does he go somewhere or grab something?
<look> actions following the reaction — does he grab anything? </look>
<assume there are some retrieved frames here>
Step 6: He grabs tape from a drawer and starts wrapping it around the toy.
Answer: 4. He tried to fix it with tape.
--------------------------------------------------

Now answer the following:

Question: {question}
Choices:
{choices}
"""
