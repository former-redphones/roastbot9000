# The Roast Bot 9000
## This was a Hackathon Project
### Authors: Bryce Tyre, Jacob Hutchens, and Oliver Thornton

## Inspiration
- It sounded fun

## What it does
The point of this project was to create something that can roast people as they walk up to your door. Nobody wants those pesky neighbors around.

## How we built it
- We used a mix of two seperate Facial recognition libraries (retinaface and backend.face_normalizer).
- We also used the Gemini API through LangChain to access an llm to create the roasts.

## Challenges we ran into
- It was a struggle to find which of the methods for AI Agent creation was the modern method. Langchain and that entire industry are growing so fast.
- It took a lot of attempts to get the Instructions for the AI to a point where it worked. At a couple of points, it was breaking its filter.
- There are a lot of facial recognition tools out there for Python, so we had to figure out which was the best to use. Once we found one that worked, we found out it was too slow, and we ended up with a hybrid approach using one fast model and one thorough model.
- We also had to figure out which TTS model worked best, and ran into issues with the engine refusing to speak more than once

## Accomplishments that we're proud of
 - The program uses multiple threads to allow seamless facial tracking while processing and speaking

## What we learned
- A LOT

## What's next for Roastbot9000
- We would like to be able to recognize even more about someone, like age or gender
- Add a compliment mode
- In the far future, Roastbot may be able to find your Facebook profile for further ammunition

- We may work on a second video with a different spin that can introduce our project in detail to new people

## How to Use it
1. Clone the Repo
2. Create a .env file in the top-level directory
3. Include GOOGLE_API_KEY='{your_api_key}' in the file with the api key you get from [here](https://aistudio.google.com/app/api-keys)
4. Follow the instructions in requirements.txt
5. You may have to change the number on line 77 in facial_recognition.py (It controls which Facecam is used [default 0])
6. You should be good to go!
