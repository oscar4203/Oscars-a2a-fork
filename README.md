# apples-to-apples-agent
Apples-to-Apples game with AI agent using various Natural Language Processing (NLP) and Machine Learning (ML) techniques.


## How to Use

1. Clone this repo to your device.
2. Go to the Google webpage https://code.google.com/archive/p/word2vec/. Download the "GoogleNews-vectors-negative300.bin.gz" word embeddings in the section "Pre-trained word and phrase vectors." Here's is the direct [Google Drive link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g).
4. Extract the compressed file into the `./apples/` directory. Do not rename the file (it should remain `GoogleNews-vectors-negative300.bin`).
5. Navigate to the root directory of the cloned repo.
6. Run the program using `python apples_to_apples.py **number players** **number of points**`. Example: `python apples_to_apples.py 3 5`.
7. The program can be run with several option flags:
    - `-A` to use all available base sets and expansion sets
    - `-V` to use the custom vector loader
    - `-P` to print all the game info and prompts in the terminal
    - `-T` to run in training mode and train the Comedian archetype
    - `-D` to enable debug mode for detailed logging
8. Have fun!
