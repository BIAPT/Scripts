# Cosine Similarity

This script was developped in the context of Comparing the motif analysis frequency of DOC patients with the average healthy participant (n = 9).

## Definition and Assumption
- This is the definition of cosine similarity we are using:
`cosine_similarity = (dot_product of a and b) / (norm of a * norm of b);`
- Posterior channels are defined as `channels_location(i).X < 0.001` meaning that every channels that are posterior to CZ including CZ are posterior.
- Anterior channels are defined as `channels_location(i).X > -0.001` meaning that every channels that are anterior to CZ including CZ are anterior.

## How it works
There is a script (calculate_cosine_similarity.m) that needs to run and will walk you through the data it need to calculate the cosine similarity.
It will call the various function that are found under the folder io, selector and math. 
Here what it does:
1. The script loads the average data and its channels location (located at /data)
2. It ask for the same information for the individual participant and loads both of these information.
3. It ask for the epoch ('EC1','IF5','EF5','EL30','EL10','EL5','EC3','EC4','EC5','EC6','EC7','EC8') to do the analysis and extract that information from the average data.
4. It ask for the frequency ('Alpha' or 'Theta') and pick that frequency band inside the average data structure.
5. It ask for the motif to compare ('M1' or ... 'M13') and extract that motif from both the averaged data and the individual participant data.
6. It normalize the motif for the individual participant.
7. It then remove the channels that are not in the individual participant data from the average data.
8. Using that subset of averaged data point it does the same thing to the individual participant data in order to make sure that both frequency count vector are of the same length.
9. It finally ask for the region ('Anterior' or 'Posterior') to do the analysis on and sift the channels from both headsets.
10. It calculate the cosine similarity on that sifted dataset.

## Future
The script work by going through a series of menu that will ask you for parameters while the it is running.
A automated version of this script is feasible, the selector needs to be removed in favor of using preset in the form of variable declared at the top of the file.
The logic to manipulate the data is inside the selector though so this need to be extracted and put inside the main scripts.

## Author
Yacine Mahdid 2019-06-26.

If something is unclear raise an issue.
