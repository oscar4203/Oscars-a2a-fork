#!/bin/bash
################################################################
#run_games.sh
#
#Used to automate playing a large number of games with the same
# players for the apples_to_apples.py program.
################################################################

if [[ $# -ne 2 ]]; then
    # echo "Usage: $0 <number_of_random_players> <number_of_ai_players> <model_type> <pretrained_models> <number_of_games>"
    echo "Usage: $0 <number_of_random_players> <number_of_ai_players>"
    exit 1
fi

NUM_RAND_PLAYERS=$1
NUM_AI_PLAYERS=$2
NUM_PLAYERS=$((NUM_AI_PLAYERS + NUM_RAND_PLAYERS))

if [[ "$NUM_PLAYERS" -lt 3 ]] || [[ "$NUM_PLAYERS" -gt 8 ]]; then
    echo "Total number of players must be between 3-8"
    exit 1
fi

read -p "Enter the model type of the AI's ('1' for Linear Regression, '2' for Neural Network): " MODEL_TYPE

if [[ $MODEL_TYPE -lt 1 ]] || [[ $MODEL_TYPE -gt 2 ]]; then
    echo "Model type must be '1' (LR) or '2' (NN)"
    exit 1
fi

read -p "How many games are to be played?: (0-?):" NUM_GAMES

if [[ $NUM_GAMES -lt 1 ]]; then
    echo "Number of games must be more than 1"
    exit 1
fi

read -p "How many point to win for each game? (1-10): " NUM_POINTS

if [[ $NUM_POINTS -lt 1 ]] || [[ $NUM_POINTS -gt 10 ]]; then
    echo "Number of points to win must be between 1-10"
    exit 1
fi

echo -e "You will now be asked what type of AI each of the $NUM_AI_PLAYERS AI players are.
        There are 3 options: Literalist, Contrarian, and Satarist.
        If you do not give an amount summing up to $NUM_AI_PLAYERS, then you will be reprompted once again."

while true; do
    read -p "Of the $NUM_AI_PLAYERS AI players, How many are Literalists?: " NUM_LITERALIST
    read -p "How many are Contrarians?: " NUM_CONTRARIAN
    read -p "How many are Satarists?: " NUM_SATARIST
    total_num_ai=$(( NUM_LITERALIST + NUM_CONTRARIAN + NUM_SATARIST ))
    if [[ "$total_num_ai" -ne $NUM_AI_PLAYERS ]]; then
        echo "Invalid Reponses. Try again."
    else
        break
    fi
done

# Function to generate player setup inputs
generate_player_inputs() {
    local num_random=$1
    local num_ai=$2
    local model_type=$3
    # local pretrained_model_type=$4
    local literalists=$4
    local contrarians=$5
    local satarists=$6
    local total_players=$((num_random + num_ai))
    local inputs=""
    for (( i=1; i<=total_players; i++ )); do
        if [ "$i" -le "$num_ai" ]; then
            inputs+="3\n"       # AI player
            inputs+="$model_type\n"
            # inputs+="$pretrained_model_type\n"
            if [[ $literalists -gt 0 ]]; then
                inputs+="1\n"
                literalists=$(( literalists - 1 ))
            elif [[ $contrarians -gt 0 ]]; then
                inputs+="2\n"
                contrarians=$(( contrarians - 1 ))
            elif [[ $satarists -gt 0 ]]; then
                inputs+="3\n"
                satarists=$(( satarists - 1 ))
            fi

        else
            inputs+="2\n"       # Random player
        fi
    done

    echo -e "$inputs"
}

#Create a temporary file to store the inputs
temp_file=$(mktemp)

# Generate player setup inputs
player_inputs=$(generate_player_inputs "$NUM_RAND_PLAYERS" "$NUM_AI_PLAYERS" "$MODEL_TYPE" "$NUM_LITERALIST" "$NUM_CONTRARIAN" "$NUM_SATARIST")
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

# Ensure the program is run from root directory
# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the root directory
cd "$SCRIPT_DIR/.."

# Run the Python program with the generated inputs
python game_driver.py $NUM_PLAYERS $NUM_POINTS < "$temp_file"

#remove the temporary file
rm -f "$temp_file"
