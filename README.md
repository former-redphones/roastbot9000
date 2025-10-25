# The Roast Bot 9000
## This was a Hackathon Project
### Authors: Bryce Tyre, Jacob Hutchens, and Oliver Thornton

## Inspiration

## What it does
The point of this project was to create something that can roast people as they walk up to your door. Nobody wants those pesky neighbhors around.

## How we built it
- We used a mix of two seperate Facial recognition libraries (retinaface and backend.face_normalizer).
- We also used the Gemini API through langchain to access an llm to create the roasts.

## Challenges we ran into
- It was a struggle to find which of the methods for AI Agent creation was the modern method. Langchain and that entire industry is growing so fast.
- It took a lot of attempts to get the Instructions for the AI to a point where it worked. At a couple of points, it was breraking its filter.
- There are a lot of facial recognition tools out there for python, so we had to figure out what was the best to use. Once we found one that worked, we found out it was too slow, and we ended up with a hybrid approach using one fast model and one thorough model.
- We also had to figure out what tts model worked best, and ran into issues with the engine refusing to speak more than once

## Accomplishments that we're proud of
 - The program uses multiple threads to allow seamless facial tracking while processing and speaking

## What we learned

## What's next for Roastbot9000
- We have plans to recognize even more about someone, like age or gender
- Add a compliment mode
- In the far future, roastbot may be able to find your facebook profile for further ammunition
- We are also working on a second video with a different spin that can introduce our project in detail to new people