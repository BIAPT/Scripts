%{
    Yacine Mahdid March 09 2020
    The purpose of this script is to find a metric to assess the difference
    between each of the matrix made in our consciousness research

%}

baseline_wpli = rand(129, 129);
anesthesia_wpli = rand(129, 129);
recovery_wpli = rand(129, 129);

diff_baseline_anesthesia = calculate_difference(baseline_wpli, anesthesia_wpli);
diff_anesthesia_recovery = calculate_difference(anesthesia_wpli, recovery_wpli);
diff_baseline_recovery = calculate_difference(baseline_wpli, recovery_wpli);

function [score] = calculate_difference(wpli_1,wpli_2)

    score = median(wpli_1(:) - wpli_2(:));

end