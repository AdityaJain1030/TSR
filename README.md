# TSR
Hackillinois Submission for 2025

## The Idea
What if we could build a robot that given any language query, could detect and retrieve an object?

## Implementation
We use MobileCLIP, a CLIP encoder-decoder by apple optimized for mobile hardware such as our robot, to process individual queries. This process runs in parallel with a video buffer, from which we encode frames on intervals using CLIP. We calculate a cosine similarity for the image across several input text queries, and if it matches one, we use computer vision to path to the object.

## Future Steps
- Pathfinding algorithm to object is very bad, maybe we can take the image encoding and then get like a similarity score across a mobilenet bounding box or something, then running something like SORT on the object bounding box
- Voice commands
- Using hardware acceleration for CLIP, need to figure out how to do this with smth like 

## How to run
Set up a virtual environment and activate it

> `/ml-mobileclip/demo.py` for the object detector demo
> `/ml-mobileclip/robot-movement.py` fo

## Hardware
Running on rasberry pi 4 with Freenove Car Motor Shield I2C. Link to full kit [here](https://store.freenove.com/products/fnk0021)