# Research Plan - David Scalais

The goal of this assignment is to create a first version of the research plan that you will execute for your Artificial Intelligence Project,
which you will continue to iterate on and refine.
To do so, you will fill out this MarkDown (md) file.

Your task is to fill out all the sections that have a [TODO] tag, following the instructions provided to guide you through each step. 
By the end of this assignment, you will have a first research plan, ready to be executed and (peer) reviewed. 

## Identifying the Problem

The first step to developing a research plan (and designing an offline evaluation experiment) is to identify the problem that we will be addressing. 
We can also refer to the problem as the evaluation objective, as it is our goal to evaluate how well our solution addresses the problem(s) we care about.

An evaluation objective is typically formulated from the perspective of one (or more) stakeholders, for example, the platform, users and item providers. Each of these stakeholders has goals, that can be achieved by designing a recommender system that has certain desired properties.

Our offline evaluation experiment then evaluates whether the solution (algorithm) has more or less of the desired properties than a representative selection of other algorithms (the baselines).
If this is the case, the solution (algorithm) is better suited to our goal than any other algorithm. 

### Stakeholders

#### Task
Answer the following questions in the context of the research questions/directions you had in mind for your project:
- Who are the stakeholders of your system?
- Which stakeholder(s) do you care about?
- Is one stakeholder more important than the others?

#### Answer

- users, producer, development team, society and editor
- Because I want my recommender to be fast (i.e. efficient) and promote relevant games, having a overall high user engagement with my platform. I care about the user, producer and development team.
- yes, I consider the stakeholders that align with my objectives to be more important. So I'm ignoring the societal and editor needs.

### Stakeholder Goals

#### Task
Now that you have listed your stakeholder(s), please answer the following questions in the context of the research questions/directions you had in mind for your project:
- What are these stakeholder(s)' goals? 
- Do these goals overlap or are they at odds with each other?
- Which stakeholder goals will you pursue in your project? 

#### Answer
- provider (making money); users (discover new games); development team (cost efficient); editor (promote his/her items); society (equal treatement to all users)
- The goals of the stakeholders that I care about overlap. So users want to discover novel item, they also want the system to work fast. This in turn will make money and align with the cost efficiency of the development team. The societal and editor are at odds, because games that they promote might not interest the users.
- as stated before: the provider, user and development team


### Desired System Properties

#### Task
Using your stakeholder goals, think about how these goals can be translated to desired properties of a recommendation algorithm. 
Please answer the following questions in the context of the research questions/directions you had in mind for your project:
- What desired properties should the algorithm have to achieve these goals?
- Can it have all these properties at the same time?

#### Answer

- efficient ranking of a large dataset, ranking novel item higher
- No immediately, but my goal is to first rank 1000 games or so with the Two tower architecture which is fast and then use another more powerfull (slower) algorithm to rank those 1000 games.


## Coming Up with a Solution

After defining the problem, and thus, the objective of our offline evaluation experiment, we can move on to designing possible solutions. 
For example, we could argue that "ItemKNN will outperform deep neural networks in a news context, because the frequency with which a recommendation model is updated is more important for its online success in the news domain than its offline ranking accuracy and ItemKNN can be updated more frequently than deep neural networks".
This is an example of a hypothesis.
The best hypotheses are rooted in theory.
This can be either a proven theory, but may also be based in observation or intuition.  
In general, any theory is preferable to no theory.
Returning to our earlier example, we could base this hypothesis on our observations of the production news recommender system at Froomle: In A/B tests, simple models that could be (and were) updated more frequently, consistently outperformed more complex models that required more training time. 
We could also base this hypothesis on a combination of proven facts and intuition: It is a proven fact that news relevance changes rapidly as news items age. As such, a recommendation algorithm that can rapidly and frequently incorporate the latest news items should outperform algorithms that cannot. Finally (and ideally), we could base this hypothesis on proven theory: Verachtert, Jeunen, and Goethals [REF] have shown that models quickly grow stale in the context of news recommendation. As such, a recommendation algorithm that lends itself to frequent and rapid updates should perform better in a news context. 

If we are unable to come up with some theory for why a hypothesis may be true, it is usually best to phrase it as a research question instead.
For example: "Is update frequency more important that offline ranking accuracy in the news domain? If so, will ItemKNN outperform deep neural networks?"

### Research Question & Hypothesis

#### Task

Please answer the following questions in the context of the evaluation objectives you had in mind for your project:
- What is my hypothesis/research question? 
- Do I have a clear hypothesis, rooted in theory, or should I pose a research question instead?
- What theory is my hypothesis based on? 
- Is this theory based on observations, intuition or is it a proven theory? 

#### Answer

- (draft) Can the Two-Tower retrieval architecture followed by a slower but more powerful reranker produce better recommendation quality than standard baselines (popularity, ItemKNN, BPR, LightGCN) while being effiecient and resulting in a lower training time?
- Todo (i cannot answer those questions yet as I Have changed my research direction)


## Designing an Offline Evaluation Experiment

Now that we have identified the problem and objectives of our evaluation, as well as defined our research question or hypothesis,
we can start designing an offline evaluation experiment that is capable of validating our hypothesis or answering our research question. 
The design of a typical offline evaluation evaluation experiment consists of seven steps, guided by two main principles.

### Principles of Offline Evaluation 

The two main principles of offline evaluation design are validity and reliability.

#### Validity 

There are different types of validity.  
In the context of your artificial intelligence project, the type of validity we care about most is internal validity: 
Ruling out alternative explanations for the results of our offline evaluation experiment.
To maximize internal validity, we have to consider how we might be unintentionally biasing our results.
There are three major threats to internal validity: confounding, selection bias and researcher bias.
Confounding is usually the result of treating the new recommendation algorithm and the baselines differently.
Selection bias is a result of differences between the sample used in the evaluation and the population it was sampled from.
The final (and most dangerous) threat is researcher bias: The fact that researchers are generally flexible regarding their experimental designs in terms of baselines, datasets, evaluation protocol, and performance measures may easily lead to researcher bias, where algorithm designers may only look for evidence that supports the superiority of their own method. 

To safeguard the internal validity of an experiment it is important to consider the following questions:
-  How might we treat the new recommendation algorithm and the other algorithms differently? 
-  How can we equalize the treatment of all algorithms under comparison (as much as possible)? 
-  Is our preprocessed evaluation dataset representative of the larger population? 
-  Are our experimental design choices in line with our research questions and hypotheses? 
-  How are we (unintentionally) biasing our results?

#### Task

Consider how you might introduce selection bias, confounding and researcher bias into your experiment, and describe
what countermeasures you will take in the design of your experiment to avoid it.

#### Answer

-
- By finetuning the baseline and chosen algorithm equally. And preprocessign the data such that it removes certain biases more geared towards certain algorithms. (e.g. popularity based algorithm) 

#### Reliability

The reliability of an offline evaluation experiment encompasses many aspects.
Firstly, it means the experiment is demonstrably and reasonably free of errors and inconsistencies. 
However, truly reilable experimental code should also be re-runnable, repeatable, reproducible, reusable, and replicable.  
An experiment is re-runnable if it is executable, or in other words, if a "restart and run all" does not produce errors -- as many JuPyTer notebooks do when executed from top to bottom.
An experiment is repeatable if it produces the same results each time it is run. 
An experiment is reproducible if someone else can take the code, run it and reobtain the reported results. For example, when your lecturers run your JuPyTer notebooks at the end of the class to see whether it produces the results you describe in your paper.
An experiment reusable when it is easy to understand and modify, thus, when it is well-documented and simple. 
An experiment is replicable when the description of the code in the report is sufficient to write the code from scratch and obtain similar results.  

#### Task

Consider how you can make your experiment reliable, re-runnable, repeatable, reproducible, reusable and replicable and describe
what measures you will take in the design of your experiment to ensure it is.

#### Answer

[TODO]


### Aspects of Offline Experimental Design

Finally, we can move on to the seven aspects of offline experimental design:
- Data Selection
- Data Preprocessing
- Data Splitting
- Selection of Baselines
- Hyperparameter Tuning
- Metric Selection

Our goal is to make experimental design decisions that will increase the validity of the result to support our evaluation objective. 

### Dataset Selection

Although the dataset has already been selected for your artificial intelligence project, it is still helpful to think about its properties
and what parts of the dataset you will use. 

#### Task 
Please answer the following questions in the context of the dataset you will use for this project:
- When was the dataset collected?
- Over which timespan?
- In which region?
- What sampling/filtering strategy was applied?
- How did users find the items in the dataset?
- What are the dataset statistics?

#### Answer

- some basic statistic: Reviews:	7,793,069, Users:	2,567,538, Items:	15,474, Bundles:	615



### Data Preprocessing

#### Task

Please answer the following questions in the context of the hypothesis/research question you had in mind for your project:
- Why do you want to filter, sample or otherwise preprocess the datasets?
- What types of filters/feature engineering do you want to apply?
- What is the best way to apply it?
- How does this filtering affect the dataset characteristics?
- How could my feature engineering introduce leakages?
- Is the sampled/filtered/processed dataset still representative of the original dataset? 

#### Answer
- So that the dataset is consitent and meaningfull for my algorithm, also to remove biased.


### Data Splitting

#### Task 
Please answer the following questions in the context of the hypothesis/research question you had in mind for your project:
- Do we have sufficient users in the datasets to split users into different sets?
- How large a timeframe span our datasets?
- Can I split it in a time-aware fashion? 
- Is the order of interactions likely important in this application domain?
- Are tehere large differences in the lengths of usrs’ interaction histories in the datasets?
- Do all users have reasonably long interaction histories, such that we may assume their preferences are well-known?
- In what way will I split the dataset? 

#### Answer

[TODO]


### Selection of Baselines

#### Task 
Please answer the following questions in the context of the hypothesis/research question you had in mind for your project:
- What other recommendation algorithms have been published recently that address the desired property and goal? 
- What general-purpose baselines have been shown to have good performance, and reasonable computational load, on this dataset, or, in this application domain?

#### Answer

- papers have use BPR as a baseline algorithm


### Hyperparameter Tuning

#### Task 
Please answer the following questions in the context of the hypothesis/research question you had in mind for your project:
- Which hyperparameters do the recommendation algorithms in this offline evaluation experiment have? 
- What is a reasonable range of values they may take, and we should explore? 
- Should we use random search or TPEs to optimize hyperparameters? 
- What target metric should we set? 
- How sensitive is the recommendation algorithm to (small) changes in the hyperparameter values?

#### Answer

[TODO] 


### Metric Selection

#### Task 
Please answer the following questions in the context of the hypothesis/research question you had in mind for your project:
- Which metric should I use to measure my desired properties?
- Does this metric truly measure the desired property related to our evaluation goal? 
- Has this metric been validated and shown to be a good proxy? 
- What will we optimize the hyperparameters of all recommendation algorithms under comparison for?

#### Answer

- in the first phase recall and in the second phase ndcg10 and history distance
- todo
- todo
- todo



## Congratulations! You’ve completed the assignment.

Please upload your research plan to GitHub Classroom.
