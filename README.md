# Kaggle Big Data Bowl

This repo is for the Kaggle competition https://www.kaggle.com/c/nfl-big-data-bowl-2020 - Kaggle Big Data Bowl of 2019.
The goal of this competition is to predict the amount of yards a player will gain by rushing the ball.

For this project I featured engineered over 100 different types of stats related to the game and created a neural network to calculate the possible running distance for the rusher.

## Brief description of the competition
- You are giving the stats of a rushing play. The stats include
	- Player specific data at the time of handoff, such as accerlation, speed, position, orientation etc.
	- Player stats such as their weight, position on the field, college they attended etc
	- Game stats such as the score of the game, field type, weather
- The goal is to predict the expected distance the rusher will gain or lose in the play
- Submissions will be evaluated on the Continuous Ranked Probability Score (CRPS). For each PlayId, you must predict a cumulative probability distribution for the yardage gained or lost. In other words, each column you predict indicates the probability that the team gains <= that many yards on the play

## Results
At the time of writing the results were 212 place out of 2038. With positions being completed over the last weeks of the 2019/2020 NFL season. 

## Pitfalls of the competition
I trusted the data too much. There was a fault in the accerlation for players in the 2017 season. This effects many of my features as that was one of the most important data pieces. This means that half of my training data was being trained with wrong data. For a newcomer to Kaggle, it's interesting to know that even their (and NFL) data can be incorrectly processed.