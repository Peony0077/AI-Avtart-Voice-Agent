# Recording a real person for a lifelike avatar

The avatar today is animated from one AI-generated photo. However good the motion, a still photo
cannot supply what makes a person look real on camera: their own skin, light, shoulders and hands,
the room around them. HeyGen and Anam build their realistic avatars from a short video of a real
person. This project can do the same (`build_avatar.py --video`): the recorded person's real face,
body and room are kept, and only the mouth is re-animated to match her speech.

## Consent first

Record only someone who agrees, in writing, that their face and likeness may be used as an AI avatar
that talks to your customers (yourself, an employee, or a hired presenter or actor with a signed
release). Do not use videos of other people found online.

## What to record

- **Length:** 1 to 2 minutes, one continuous take.
- **Framing:** head and shoulders, face in the upper middle, looking into the camera lens (like a
  video call). Leave some space above the head.
- **Camera:** on a tripod or a stable stand, not hand-held. A phone's main (back) camera is fine.
  1080p, 25 or 30 frames per second.
- **Light:** soft and even on the face, such as a window in front of the person or a ring light. No
  strong light from behind.
- **Place:** the real setting you want customers to see, such as an office, a shop or a tidy room.
  Nothing should move in the background (no screens, fans or people walking past).
- **Clothes:** plain colors, no fine stripes.

## How the person should behave

Act like a friendly listener in a video call, **with the mouth closed most of the time** (the mouth
is re-animated for speech):

- look at the camera, breathe and blink naturally
- small natural head movements: a nod now and then, a slight tilt, glancing aside and back
- a warm, natural smile now and then, and a few seconds of a "thoughtful" look
- relaxed shoulders. Hands may stay out of the picture.

Avoid talking, laughing with an open mouth, covering the face, or leaving the frame.

## Then

Copy the video to `data\avatars\<name>\video.mp4` and build the avatar from it (the avatar venv,
see the README):

```powershell
avatar\musetalk\.venv\Scripts\python avatar\musetalk\build_avatar.py --video data\avatars\<name>\video.mp4 --id <name>
```

(Linux: `avatar/musetalk/.venv/bin/python avatar/musetalk/build_avatar.py --video data/avatars/<name>/video.mp4 --id <name>`.)
The build takes about an hour on the RTX 3060 (a full photo build with all head gestures and body
motion: ~80 min), less on a faster GPU. Then set
`avatar.avatar_id: <name>` in `config\profiles\avatar.yaml` to replace the photo avatar.

The build's face cropper uses InsightFace models that are licensed for non-commercial use only; see
"Licences and commercial use" in the README before using the avatar for a business.
