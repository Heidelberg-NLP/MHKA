
In this work, we define a new Counterfactual Invariance Prediction (CIP) task that tests the capability of models to predict whether under the assumption of a counterfactual event, a (later) factual event remains invariant or not in a narrative context.


The formal setup is: given the first three consecutive sentences from a narrative story s1 (premise), s2 (initial context), s3 (factual event) and an additional sentence s0 that is counterfactual to the initial context s2, the task is to predict whether s3 is invariant given s1, s0 or not. The train/dev/test
data are balanced with an equal number of Yes/No answers, hence the random baseline is 50%. To compute human performance, we gave 100 instances from the test set to expert evaluators. Human accuracy on the CIP task is at 84.8%
