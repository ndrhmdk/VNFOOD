API_KEY = 'AIzaSyDoMjd8WEokmnSxW1QTRWEWV2Vq5z1Xp4k'


GEMINI_INSTRUCTION = """
    From the given image, please describe the ingredients that you see and return them in a valid json format as followed:
    {
        "ingredients" : ['beef', 'pepper', 'egg']
    }
    Remember not to make up any ingredients that you don't find in the image
"""