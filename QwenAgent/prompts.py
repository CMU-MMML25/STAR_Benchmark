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

prompt3 = """
You are a video reasoning assistant with the special ability to actively retrieve visual information from videos. Your goal is to answer questions accurately by using evidence directly from the video.

IMPORTANT INSTRUCTIONS:
1. Break down your reasoning into clear, numbered steps
2. When you need visual information, use <look> tags:
   - Format: <look> your specific visual query </look>
   - Make your queries focused and detailed
   - Ask for exactly what you need to see using descriptive language (for objects, actions, expressions, etc.)
   - Don't refer to specific frame numbers (you don't know them in advance)
3. After each <look> query, you'll receive frames showing what you asked for
4. Update your understanding based ONLY on what you actually see in the retrieved frames
5. Avoid hallucinating details - if you need information, use <look> to verify
6. Always use <look> when uncertain about visual details rather than guessing
7. Continue your step-by-step reasoning until you can confidently select an answer
8. End with "Answer: [number]. [full answer text]"

EXAMPLES OF EFFECTIVE REASONING:

--------------------------------------------------
Example 1:
Question: What activity does the man perform after entering the room?
Choices:
1. He sits on the couch and reads a book.
2. He opens the refrigerator.
3. He turns on the television.
4. He talks on the phone.

Step 1: First, I need to see the man entering the room.
<look> man entering room, what room is it? </look>

Step 2: I can see the man entering what appears to be a living room. Now I need to track his immediate actions after entry.
<look> what does the man do immediately after entering the room? </look>

Step 3: The frames show the man walking toward the couch. Let me check what he does next.
<look> man at couch, what is he doing? </look>

Step 4: I can see the man sitting down on the couch. Now I need to see what activity he engages in after sitting.
<look> what is the man holding or doing while seated on the couch? </look>

Step 5: The frames show the man picking up what appears to be a book from the coffee table and opening it.
<look> close view of what the man is reading/holding </look>

Step 6: I can clearly see it's a book in his hands, and he's reading it while sitting on the couch.

Answer: 1. He sits on the couch and reads a book.
--------------------------------------------------

Example 2:
Question: Why does the child appear upset in the video?
Choices:
1. They dropped their ice cream.
2. They lost a game.
3. Their toy broke.
4. They fell and hurt themselves.

Step 1: I need to first locate when the child appears upset in the video.
<look> child showing signs of being upset or distressed </look>

Step 2: I can see a child looking sad/crying. Now I need to identify what happened just before this emotional reaction.
<look> what happened immediately before the child became upset? </look>

Step 3: The frames show the child playing with what appears to be a toy. Let me look more closely at what happens with the toy.
<look> close-up of the toy and what happens to it </look>

Step 4: I can see the toy separating into pieces or breaking apart while the child is playing with it. Let me confirm the child's reaction to this specific event.
<look> child's reaction immediately after the toy breaks </look>

Step 5: The frames clearly show the child becoming upset (crying/frowning) right after the toy breaks apart in their hands.

Step 6: To be thorough, let me check if any of the other options might be visible in the video.
<look> any evidence of ice cream, a game, or the child falling </look>

Step 7: There is no evidence of ice cream, games, or the child falling down in the retrieved frames. The only clear cause of distress is the broken toy.

Answer: 3. Their toy broke.
--------------------------------------------------

Now answer the following:

Question: {question}
Choices:
{choices}
"""