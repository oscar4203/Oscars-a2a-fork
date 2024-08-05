#!/bin/bash
################################################################
#run_games.sh
#
#Used to automate playing a large number of games with the same
# players for the apples_to_apples.py program.
################################################################

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 <number_of_random_players> <number_of_ai_players> <model_type> <pretrained_models> <number_of_games>"
    exit 1
fi 

NUM_RAND_PLAYERS=$1
NUM_AI_PLAYERS=$2
MODEL_TYPE=$3
PRETRAINED_MODEL_TYPE=$4
NUM_GAMES=$5
NUM_PLAYERS=$((NUM_AI_PLAYERS + NUM_RAND_PLAYERS))

if [[ $MODEL_TYPE -lt 1 ]] || [[ $MODEL_TYPE -gt 2 ]]; then
    echo "Model type must be 1 (LR) or 2 (NN)"
    exit 1
fi

if [[ $PRETRAINED_MODEL_TYPE -lt 1 ]] || [[ $PRETRAINED_MODEL_TYPE -gt 3 ]]; then
    echo "Pretrained model type must be 1 (Literalist), 2 (Contrarian), or 3 (Satarist)"
    exit 1
fi

if [[ "$NUM_PLAYERS" -lt 3 ]] || [[ "$NUM_PLAYERS" -gt 8 ]]; then
    echo "Total number of players must be between 3-8"
    exit 1
fi

# Function to generate player setup inputs
generate_player_inputs() {
    local num_random=$1
    local num_ai=$2
    local model_type=$3
    local pretrained_model_type=$4
    local total_players=$((num_random + num_ai))
    local inputs=""
    for (( i=1; i<=total_players; i++ )); do
        if [ "$i" -le "$num_ai" ]; then
            inputs+="3\n"       # AI player
            inputs+="$model_type\n"
            inputs+="$pretrained_model_type\n"
        else
            inputs+="2\n"       # Random player
        fi
    done

    echo -e "$inputs"
}

#Create a temporary file to store the inputs
temp_file=$(mktemp)

# Generate player setup inputs
player_inputs=$(generate_player_inputs "$NUM_RAND_PLAYERS" "$NUM_AI_PLAYERS" "$MODEL_TYPE" "$PRETRAINED_MODEL_TYPE")
game_inputs="${player_inputs}\n"

# loop that adds judge & option inputs per the number of games to be played
for (( game=1; game<=NUM_GAMES; game++ )); do
    # Determine the starting judge for the current game
    judge=$(( ((game - 1) % (NUM_RAND_PLAYERS + NUM_AI_PLAYERS)) + 1 ))

    # Creates input sequence for cycling through judges game by game
    game_inputs+="${judge}\n"

    # adds the input for restarting the game with the same player, given it's not the last game
    if [[ "$game" -ne "$NUM_GAMES" ]]; then
        game_inputs+="1\n"
    fi
    
done

#Ends the session of games. 
game_inputs+="3\n"

# Write the inputs to the temporary file
echo -e "$game_inputs" > "$temp_file"   

# Run the Python program with the generated inputs
python apples_to_apples.py $NUM_PLAYERS 3 < "$temp_file"

#remove the temporary file
rm -f "$temp_file"