# Introduction to Azure Machine Learning


Microsoft Scholarship Foundation course Nanodegree Program



1 Welcome
1.1 Welcome to Udacity

https://photos.app.goo.gl/N1oaAg1LygquueEx7

1.2 How this program works

Welcome and Congratulations!
Udacity and Microsoft are excited to welcome you to the Introduction to Machine Learning on Azure. Here are some helpful details as you start the program.
How the Scholarship Foundation works:
The lessons in this course have been designed to introduce you to Machine Learning capabilities using Azure. We recommend you take the time to go through the carefully prepared lessons, as well as the labs which are required for course completion to qualify for a Nanodegree scholarship. If you have questions, we encourage you to ask directly in our student community.

1.3 Join the Scholarship Community

We’ve created a custom Slack workspace for your course classmates and you.
In this exclusive student community, experts from the Udacity team will be able to answer your questions and provide technical advice. What’s more, you can use the Slack workspace to collaborate with and get support from classmates or ask questions to our team of Community Managers. Below is the auto-invite link to join the Slack community:
Join Slack community here
If you’re new to Slack, don’t worry. You’ll find it’s super easy to learn and that it’s a great way to communicate with other students.
Orientation
Our Community Managers will also be hosting an orientation to help you get started. We encourage you to attend.
When: Thursday, Jul 9th, 2020 @ 10 AM PST / 5 PM GMT. Follow this link to attend the Orientation:
Scholarship Google Site
The Microsoft Azure Challenge Google site will be your go-to spot for all information on this Scholarship program. Check back often as updates are made regularly. If you have a question, the answer is most likely here!
Link: https://bit.ly/microsoft-azure-challenge
Happy learning!

1.4 Why Machine Learning?

https://photos.app.goo.gl/gKALzTVPDeU89VhZ8

1.5 Course Overview

https://photos.app.goo.gl/xviroJ7t2JZ6Wo1T7

Machine Learning in Context
In this course, we'll provide you with an introduction to the fantastic world of machine learning (ML). We will first help you understand the overall place of machine learning in the broader context of computer science. We'll take you through its history, perspectives, approaches, challenges, essential tools, and fundamental processes.

Training a Model
After that, we will focus on the core machine learning process: training a model. We'll cover the entire chain of tasks, from data import, transformation, and management to training, validating, and evaluating the model.

Foundational Concepts
Since this is an introductory course, our goal will be to give you an understanding of foundational concepts. We'll talk about the fundamentals of supervised (classification and regression) and unsupervised (clustering) approaches.

More Advanced Techniques
Then, building on these basics, we'll have a look at more advanced techniques like ensemble learning and deep learning.

Classic Applications of ML
We'll also cover some of the best known specific applications of machine learning, like recommendations, text classification, anomaly detection, forecasting, and feature learning.

Managed Services, Cloud Computing, and Microsoft Azure
Many machine learning problems involve substantial requirements—things like model management, computational resource allocation, and operationalization. Meeting all these requirements on your own can be difficult and inefficient—which is why it's often very beneficial to use Software as a Service (SaaS), managed services, and cloud computing to outsource some of the work. And that's exactly what we'll be doing in this course—specifically, we'll show you how to leverage Microsoft Azure to empower your machine learning solutions.


Responsible AI
At the very end of the course, we'll talk about the broader impact of machine learning. We'll discuss some of the challenges and risks that are involved with machine learning, and then see how we can use principles of responsible artificial intelligence, such as transparency and explainability, to help ensure our machine learning applications generate positive impact, and avoid harming others.

2 Introduction to Machine Learning

2.1 Lesson Overview

https://photos.app.goo.gl/k7YtNbW37JdEzxXTA

In this lesson, our goal is to give you a high-level introduction to the field of machine learning, including the broader context in which this branch of computer science exists.

Here are the main topics we'll cover:

•	What machine learning is and why it's so important in today's world
•	The historical context of machine learning
•	The data science process
•	The types of data that machine learning deals with
•	The two main perspectives in ML: the statistical perspective and the computer science perspective
•	The essential tools needed for designing and training machine learning models
•	The basics of Azure ML
•	The distinction between models and algorithms
•	The basics of a linear regression model
•	The distinction between parametric vs. non-parametric functions
•	The distinction between classical machine learning vs. deep learning
•	The main approaches to machine learning
•	The trade-offs that come up when making decisions about how to design and training machine learning models

In the process, you will also train your first machine learning model using Azure Machine Learning Studio.




2.2 What is Machine Learning?

One of our goals in this lesson is to help you get a clearer, more specific understanding of what machine learning is and how it differs from other approaches.

Let's start with a classic definition. If you look up the term in a search engine, you might find something like this:
Machine learning is a data science technique used to extract patterns from data, allowing computers to identify related data, and forecast future outcomes, behaviors, and trends.

Let's break that down a little. One important component of machine learning is that we are taking some data and using it to make predictions or identify important relationships. But looking for patterns in data is done in traditional data science as well. So how does machine learning differ? In this next video, we'll go over a few examples to illustrate the difference between machine learning and traditional programming.

https://photos.app.goo.gl/C4s6MZQdopq3pXpN9

    
 

2.3 Applications of Machine Learning

The applications of machine learning are extremely broad! And the opportunities cut across industry verticals. Whether the industry is healthcare, finance, manufacturing, retail, government, or education, there is enormous potential to apply machine learning to solve problems in more efficient and impactful ways.

We'll take a tour through some of the major areas where machine learning is applied, mainly just to give you an idea of the scope and type of problems that machine learning is commonly used to address.

https://photos.app.goo.gl/VqJ4TAooi62PyFsy8

Examples of Applied Machine Learning

Machine learning is used to solve an extremely diverse range of problems. For your reference, here are all the examples we discussed in the video, along with links to further reading in case you are curious and want to learn more about any of them:


Automate the recognition of disease

Trained physicians can only review and evaluate a limited volume of patients or patient images (X-rays, sonograms, etc.). Machine learning can be used to spot the disease, hence reducing physician burnout. For example, Google has trained a deep learning model to detect breast cancer and Stanford researchers have used deep learning models to diagnose skin cancer.

Recommend next best actions for individual care plans

With the mass digitization of patient data via systems that use EMRs (Electronic Medical Records) and EHRs (Electronic Health Records), machine learning can be used to help build effective individual care plans. For example, IBM Watson Oncology can help clinicians explore potential treatment options. More examples of how machine learning impacts healthcare can be found here.

Enable personalized, real-time banking experiences with chatbots

You've likely encountered this when you call a customer service number. Machine learning can be used to intercept and handle common, straightforward issues through chat and messaging services, so customers can quickly and independently resolve simple issues that would otherwise have required human intervention. With the chatbot, a customer can simply type in a question and the bot engages to surface the answer. Refer to this article to find more information about chatbot powered machine learning.

Identify the next best action for the customer

Real-time insights that incorporate machine learning tools—such as sentiment analysis—can help organizations assess the likelihood of a deal closing or the level of a customer’s loyalty. Personally-tailored recommendations powered by machine learning can engage and delight customers with information and offers that are relevant to them.

Capture, prioritize, and route service requests to the correct employee, and improve response times

A busy government organization gets innumerable service requests on an annual basis. Machine learning tools can help to capture incoming service requests, to route them to the correct employee in real-time, to refine prioritization, and improve response times. Can check out this article if you're curious to learn more about ticket routing






2.4 Brief History of Machine Learning

https://photos.app.goo.gl/sWsPMe35Bx7nemTN6


 

Further Reading

•	What’s the Difference Between Artificial Intelligence, Machine Learning and Deep Learning? by Michael Copeland at NVIDIA




2.5 The Data Science Process

Big data has become part of the lexicon of organizations worldwide, as more and more organizations look to leverage data to drive informed business decisions. With this evolution in business decision-making, the amount of raw data collected, along with the number and diversity of data sources, is growing at an astounding rate. This data presents enormous potential.
Raw data, however, is often noisy and unreliable and may contain missing values and outliers. Using such data for modeling can produce misleading results. For the data scientist, the ability to combine large, disparate data sets into a format more appropriate for analysis is an increasingly crucial skill.

The data science process typically starts with collecting and preparing the data before moving on to training, evaluating, and deploying a model. Let's have a look.

https://photos.app.goo.gl/fcPS9KoMaDnY4Bs9A
   

2.6 Common Types of Data

https://photos.app.goo.gl/Wd91jBEwrXqmjyLx9

It's All Numerical in the End
Note that although we've described numerical data as a distinct category, it is actually involved in some way with all of the data types we've described. With the example of stock performance (above) the stock prices are numerical data points. So why do we give this as an example of "time-series data" rather than "numerical data"? It is the ordering of the numerical data points across points in time that leads us to call the data time-series data.

What is more, all data in machine learning eventually ends up being numerical, regardless of whether it is numerical in its original form, so it can be processed by machine learning algorithms.

For example, we may want to use gender information in the dataset to predict if an individual has heart disease. Before we can use this information with a machine learning algorithm, we need to transfer male vs. female into numbers, for instance, 1 means a person is male and 2 means a person is female, so it can be processed. Note here that the value 1 or 2 does not carry any meaning.

Another example would be using pictures uploaded by customers to identify if they are satisfied with the service. Pictures are not initially in numerical form but they will need to be transformed into RGB values, a set of numerical values ranging from 0 to 255, to be processed.  


2.7 Tabular Data

In machine learning, the most common type of data you'll encounter is tabular data—that is, data that is arranged in a data table. This is essentially the same format as you work with when you look at data in a spreadsheet.
Here's an example of tabular data showing some different clothing products and their properties:


SKU	Make	Color	Quantity	Price
908721	Guess	Blue	789	45.33
456552	Tillys	Red	244	22.91
789921	A&F	Green	387	25.92
872266	Guess	Blue	154	17.56

Notice how tabular data is arranged in rows and columns.

 


https://photos.app.goo.gl/bMD2soXPGr54LUQB7


 
https://photos.app.goo.gl/qTvMKcK1GSHzfPZ88

Vectors
It is important to know that in machine learning we ultimately always work with numbers or specifically vectors.
A vector is simply an array of numbers, such as (1, 2, 3)—or a nested array that contains other arrays of numbers, such as (1, 2, (1, 2, 3)).
Vectors are used heavily in machine learning. If you have taken a basic course in linear algebra, then you are probably in good shape to begin learning about how they are used in machine learning. But if linear algebra and vectors are totally new to you, there are some great free resources available to help you learn. You may want to have a look at Khan Academy's excellent introduction to the topic here or check out Udacity's free Linear Algebra Refresher Course.
For now, the main points you need to be aware of are that:
•	All non-numerical data types (such as images, text, and categories) must eventually be represented as numbers
•	In machine learning, the numerical representation will be in the form of an array of numbers—that is, a vector
As we go through this course, we'll look at some different ways to take non-numerical data and vectorize it (that is, transform it into vector form).
2.8 Scaling Data

Scaling data means transforming it so that the values fit within some range or scale, such as 0–100 or 0–1. There are a number of reasons why it is a good idea to scale your data before feeding it into a machine learning algorithm.
Let's consider an example. Imagine you have an image represented as a set of RGB values ranging from 0 to 255. We can scale the range of the values from 0–255 down to a range of 0–1. This scaling process will not affect the algorithm output since every value is scaled in the same way. But it can speed up the training process, because now the algorithm only needs to handle numbers less than or equal to 1.
Two common approaches to scaling data include standardization and normalization.

https://photos.app.goo.gl/Kdiy228JfncHhwkT9

Standardization

Standardization rescales data so that it has a mean of 0 and a standard deviation of 1.
The formula for this is:

(𝑥 − 𝜇) /𝜎

We subtract the mean (𝜇) from each value (x) and then divide by the standard deviation (𝜎). To understand why this works, it helps to look at an example. Suppose that we have a sample that contains three data points with the following values:
50  
100  
150  

The mean of our data would be 100, while the sample standard deviation would be 50.
Let's try standardizing each of these data points. The calculations are:

(50 − 100)/50 = -50/50 = -1
(100 − 100)/50 = 0/50 = 0
(150 − 100)/50 = 50/50 = 1

Thus, our transformed data points are:
-1  
0  
1
Again, the result of the standardization is that our data distribution now has a mean of 0 and a standard deviation of 1.

Normalization

Normalization rescales the data into the range [0, 1].

The formula for this is:
(𝑥 −𝑥𝑚𝑖𝑛)/(𝑥𝑚𝑎𝑥 −𝑥𝑚𝑖𝑛)

For each individual value, you subtract the minimum value (𝑥𝑚𝑖𝑛) for that input in the training dataset, and then divide by the range of the values in the training dataset. The range of the values is the difference between the maximum value (𝑥𝑚𝑎𝑥) and the minimum value (𝑥𝑚𝑖𝑛).

Let's try working through an example with those same three data points:

50  
100  
150  

The minimum value (𝑥𝑚𝑖𝑛) is 50, while the maximum value (𝑥𝑚𝑎𝑥) is 150. The range of the values is 𝑥𝑚𝑎𝑥 −𝑥𝑚𝑖𝑛 = 150 − 50 = 100.
Plugging everything into the formula, we get:

(50 − 50)/100 = 0/100 = 0
(100 − 50)/100 = 50/100 = 0.5
(150 − 50)/100 = 100/100 = 1

Thus, our transformed data points are:

0
0.5  
1

Again, the goal was to rescale our data into values ranging from 0 to 1—and as you can see, that's exactly what the formula did.
 

 

 





2.9 Encoding Categorical Data

As we've mentioned a few times now, machine learning algorithms need to have data in numerical form. Thus, when we have categorical data, we need to encode it in some way so that it is represented numerically.

There are two common approaches for encoding categorical data: ordinal encoding and one hot encoding.

https://photos.app.goo.gl/iQEM3wgzNbZbbo7HA

Ordinal Encoding

In ordinal encoding, we simply convert the categorical data into integer codes ranging from 0 to (number of categories – 1). Let's look again at our example table of clothing products:

SKU	Make	Color	Quantity	Price
908721	Guess	Blue	789	45.33
456552	Tillys	Red	244	22.91
789921	A&F	Green	387	25.92
872266	Guess	Blue	154	17.56

If we apply ordinal encoding to the Make property, we get the following:
Make	Encoding
A&F	0
Guess	1
Tillys	2

And if we apply it to the Color property, we get:

Color	Encoding
Red	0
Green	1
Blue	2

Using the above encoding, the transformed table is shown below:

SKU	Make	Color	Quantity	Price
908721	1	2	789	45.33
456552	2	0	244	22.91
789921	0	1	387	25.92
872266	1	2	154	17.56

One of the potential drawbacks to this approach is that it implicitly assumes an order across the categories. In the above example, Blue (which is encoded with a value of 2) seems to be more than Red (which is encoded with a value of 1), even though this is in fact not a meaningful way of comparing those values. This is not necessarily a problem, but it is a reason to be cautious in terms of how the encoded data is used.







One-Hot Encoding

One-hot encoding is a very different approach. In one-hot encoding, we transform each categorical value into a column. If there are n categorical values, n new columns are added. For example, the Color property has three categorical values: Red, Green, and Blue, so three new columns Red, Green, and Blue are added.

If an item belongs to a category, the column representing that category gets the value 1, and all other columns get the value 0. For example, item 908721 (first row in the table) has the color blue, so we put 1 into that Blue column for 908721 and 0 into the Red and Green columns. Item 456552 (second row in the table) has color red, so we put 1 into that Red column for 456552 and 0 into the Green and Blue columns.

If we do the same thing for the Make property, our table can be transformed as follows:

SKU	A&F	Guess	Tillys	Red	Green	Blue	Quantity	Price
908721	0	1	0	0	0	1	789	45.33
456552	0	0	1	1	0	0	244	22.91
789921	1	0	0	0	1	0	387	25.92
872266	0	1	0	0	0	1	154	17.56

One drawback of one-hot encoding is that it can potentially generate a very large number of columns.

 
 

 

 

2.10 Image Data

Images are another example of a data type that is commonly used as input in machine learning problems—but that isn't initially in numerical format. So, how do we represent an image as numbers? Let's have a look.

https://photos.app.goo.gl/WNmroPP3qjLWS6gV8

Taking a Closer Look at Image Data
Let's look a little closer at how an image can be encoded numerically. If you zoom in on an image far enough, you can see that it consists of small tiles, called pixels:
 

The color of each pixel is represented with a set of values:
•	In grayscale images, each pixel can be represented by a single number, which typically ranges from 0 to 255. This value determines how dark the pixel appears (e.g., 0 is black, while 255 is bright white).

•	In colored images, each pixel can be represented by a vector of three numbers (each ranging from 0 to 255) for the three primary color channels: red, green, and blue. These three red, green, and blue (RGB) values are used together to decide the color of that pixel. For example, purple might be represented as 128, 0, 128 (a mix of moderately intense red and blue, with no green).
The number of channels required to represent the color is known as the color depth or simply depth. With an RGB image, depth = 3, because there are three channels (Red, Green, and Blue). In contrast, a grayscale image has depth = 1, because there is only one channel.

Encoding an Image
Let's now talk about how we can use this data to encode an image. We need to know the following three things about an image to reproduce it:
•	Horizontal position of each pixel
•	Vertical position of each pixel
•	Color of each pixel
Thus, we can fully encode an image numerically by using a vector with three dimensions. The size of the vector required for any given image would be the height * width * depth of that image.
 

 

Other Preprocessing Steps

In addition to encoding an image numerically, we may also need to do some other preprocessing steps. Generally, we would want to ensure that the input images have a uniform aspect ratio (e.g., by making sure all of the input images are square in shape) and are normalized (e.g. subtract mean pixel value in a channel from each pixel value in that channel). Some other preprocessing operations we might want to do to clean the input images include rotation, cropping, resizing, denoising, and centering the image.

2.11 Text Data

Text is another example of a data type that is initially non-numerical and that must be processed before it can be fed into a machine learning algorithm. Let's have a look at some of the common tasks we might do as part of this processing.





Normalization

One of the challenges that can come up in text analysis is that there are often multiple forms that mean the same thing. For example, the verb to be may show up as is, am, are, and so on. Or a document may contain alternative spellings of a word, such as behavior vs. behaviour. So one step that you will sometimes conduct in processing text is normalization.

Text normalization is the process of transforming a piece of text into a canonical (official) form.
Lemmatization is an example of normalization. A lemma is the dictionary form of a word and lemmatization is the process of reducing multiple inflections to that single dictionary form. For example, we can apply this to the is, am, are example we mentioned above:
Original word	Lemmatized word
is	be
are	be
am	be
In many cases, you may also want to remove stop words. Stop words are high-frequency words that are unnecessary (or unwanted) during the analysis. For example, when you enter a query like which cookbook has the best pancake recipe into a search engine, the words which and the are far less relevant than cookbook, pancake, and recipe. In this context, we might want to consider which and the to be stop words and remove them prior to analysis.

Here's another example:
Original text	Normalized text
The quick fox.	[quick, fox]
The lazzy dog.	[lazy, dog]
The rabid hare.	[rabid, hare]
Here we have tokenized the text (i.e., split each string of text into a list of smaller parts or tokens), removed stop words (the), and standardized spelling (changing lazzy to lazy).
 

Vectorization
After we have normalized the text, we can take the next step of actually encoding it in a numerical form. The goal here is to identify the particular features of the text that will be relevant to us for the particular task we want to perform—and then get those features extracted in a numerical form that is accessible to the machine learning algorithm. Typically this is done by text vectorization—that is, by turning a piece of text into a vector. Remember, a vector is simply an array of numbers—so there are many different ways that we can vectorize a word or a sentence, depending on how we want to use it. Common approaches include:
•	Term Frequency-Inverse Document Frequency (TF-IDF) vectorization
•	Word embedding, as done with Word2vec or Global Vectors (GloVe)
The details of these approaches are a bit outside the scope of this class, but let's take a closer look at TF-IDF as an example. The approach of TF-IDF is to give less importance to words that contain less information and are common in documents, such as "the" and "this"—and to give higher importance to words that contain relevant information and appear less frequently. Thus TF-IDF assigns weights to words that signify their relevance in the documents.
Here's what the word importance might look like if we apply it to our example

	quick	fox	lazy	dog	rabid	hare	the
	0.32	0.23	0.12	0.23	0.56	0.12	0.0
Here's what that might look like if we apply it to the normalized text:
	quick	fox	lazy	dog	rabid	hare
[quick, fox]	0.32	0.23	0.0	0.0	0.0	0.0
[lazy, dog]	0.0	0.0	0.12	0.23	0.0	0.0
[rabid, hare]	0.0	0.0	0.0	0.0	0.56	0.12

Noticed that "the" is removed since it has 0 importance here.
Each chunk of text gets a vector (represented here as a row in the table) that is the length of the total number of words that we are interested in (in this case, six words). If the normalized text does not have the word in question, then the value in that position is 0, whereas if it does have the word in question, it gets assigned to the importance of the word.
 
 

Feature Extraction
As we talked about earlier, the text in the example can be represented by vectors with length 6 since there are 6 words total.
[quick, fox] as (0.32, 0.23, 0.0, 0.0, 0.0, 0.0)
[lazy, dog] as (0.0, 0.0, 0.12, 0.23, 0.0, 0.0)
[rabid, hare] as (0.0, 0.0, 0.0, 0.0, 0.56, 0.12)
We understand the text because each word has a meaning. But how do algorithms understand the text using the vectors, in other words, how do algorithms extract features from the vectors?
Vectors with length n can be visualized as a line in an n dimension space. For example, a vector (1,1) can be viewed as a line starting from (0, 0) and ending at (1,1).

 
Any vector with the same length can be visualized in the same space. How close one vector is to another can be calculated as vector distance. If two vectors are close to each other, we can say the text represented by the two vectors have a similar meaning or have some connections. For example, if we add [lazy, fox] to our example:

	quick	fox	lazy	dog	rabid	hare
[quick, fox]	0.32	0.23	0.0	0.0	0.0	0.0
[lazy, dog]	0.0	0.0	0.12	0.23	0.0	0.0
[rabid, hare]	0.0	0.0	0.0	0.0	0.56	0.12
[lazy, fox]	0.0	0.23	0.12	0.0	0.0	0.0

Apparently, [lazy, fox] is more similar to [lazy, dog] than [rabid, hare], so the vector distance of [lazy, fox] and [lazy, dog] is smaller than that to [lazy, fox] and [rabid, hare].

 



The Whole Pipeline
In this next video, we'll first review the above steps—normalization and vectorization—and then talk about how they fit into the larger goal of training a machine-learning model to analyze text data.

https://photos.app.goo.gl/N6ot63Qfd9TRYRUE9

In summary, a typical pipeline for text data begins by pre-processing or normalizing the text. This step typically includes tasks such as breaking the text into sentence and word tokens, standardizing the spelling of words, and removing overly common words (called stop words).

The next step is feature extraction and vectorization, which creates a numeric representation of the documents. Common approaches include TF-IDF vectorization, Word2vec, and Global Vectors (GloVe).
Last, we will feed the vectorized document and labels into a model and start the training.
 
 
2.12 Two Perspectives on ML

https://photos.app.goo.gl/dhtiGiNoLU7Upuca9

Computer science vs. Statistical perspective

As you can see, data plays a central role in how problems are modeled in machine learning. In very broad terms, we can think of machine learning as a matter of using some data (perhaps historical data that we already have on hand) to train a model. Then, once the model is trained, we can feed it new input data and have it tell us something useful.

So the general idea is that we create models and then feed data into these models to generate outputs. These outputs might be, for example, predictions for future trends or patterns in the data.

This idea draws on work not only from computer science, but also statistics—and as a result, you will often see the same underlying machine learning concepts described using different terms. For example, a computer scientist might say something like:
We are using input features to create a program that can generate the desired output.

In contrast, someone with a background in statistics might be inclined to say something more like:

We are trying to find a mathematical function that, given the values of the independent variables can predict the values of the dependent variables.

While the terminology are different, the challenges are the same, that is how to get the best possible outcome.
 

In the end, having an understanding of the underlying concepts is more important than memorizing the terms used to describe those concepts. However, it's still essential to be familiar with the terminology so that you don't get confused when talking with people from different backgrounds.

Over the next couple of pages, we'll take a look at these two different perspectives and get familiar with some of the related terminology.






2.13 The Computer Science Perspective

https://photos.app.goo.gl/RPEzhXaG3yyzedNq7

Computer science terminology

As we discussed earlier, one of the simplest ways we can organize data for machine learning is in a table, like the table of clothing products we looked at earlier in this lesson:

SKU	Make	Color	Quantity	Price
908721	Guess	Blue	789	45.33
456552	Tillys	Red	244	22.91
789921	A&F	Green	387	25.92
872266	Guess	Blue	154	17.56

What are some of the terms we can use to describe this data?
For the rows in the table, we might call each row an entity or an observation about an entity. In our example above, each entity is simply a product, and when we speak of an observation, we are simply referring to the data collected about a given product. You'll also sometimes see a row of data referred to as an instance, in the sense that a row may be considered a single example (or instance) of data.

For the columns in the table, we might refer to each column as a feature or attribute which describes the property of an entity. In the above example, color and quantity are features (or attributes) of the products.






Input and output

Remember that in a typical case of machine learning, you have some kind of input which you feed into the machine learning algorithm, and the algorithm produces some output. In most cases, there are multiple pieces of data being used as input. For example, we can think of a single row from the above table as a vector of data points:

(908721, Guess, Blue, 789, 45.33)


Again, in computer science terminology, each element of the input vector (such as Guess or Blue) is referred to as an attribute or feature. Thus, we might feed these input features into our machine learning program and the program would then generate some kind of desired output (such as a prediction about how well the product will sell). This can be represented as:


Output = Program(Input Features)

An important step in preparing your data for machine learning is extracting the relevant features from the raw data. (The topic of feature extraction is an important one that we'll dive into in greater detail in a later lesson.)
 

 






2.14 The Statistical Perspective

https://photos.app.goo.gl/oMiyrdM5gQhv7mhx7

Statistical terminology

In statistics, you'll also see the data described in terms of independent variables and dependent variables. These names come from the idea that the value of one variable may depend on the value of some other variables. For example, the selling price of a house is the dependent variable that depends on some independent variables—like the house's location and size.
In the example of clothing products we looked at earlier in this lesson:

SKU	Make	Color	Quantity	Price
908721	Guess	Blue	789	45.33
456552	Tillys	Red	244	22.91
789921	A&F	Green	387	25.92
872266	Guess	Blue	154	17.56

We might use data in each row (e.g. (908721, Guess, Blue, 789, 45.33)) to predict the sale of the corresponding item. Thus, the sale of each item is dependent on the data in each row. We can call the data in each row the independent variables and call the sale the dependent variable.










Input and output

From a statistical perspective, the machine learning algorithm is trying to learn a hypothetical function (f) such that:

Output Variable = f(Input Variables)

Typically, the independent variables are the input, and the dependent variables are the output. Thus, the above formula can also be expressed as:

Dependent Variable = f(Independent Variables)

In other words, we are feeding the independent variables into the function, and the function is giving us the resulting values of the dependent variables. With the housing example, we might want to have a function that can take the independent variables of size and location as input and use these to predict the likely selling price of the house as output.

Yet another way to represent this concept is to use shorthand notation. Often, the input variables are denoted as X and the output variable is denoted as Y:

Y = f(X)

In the case of multiple input variables, X would be an input vector, meaning that it would be composed of multiple individual inputs (e.g. (908721, Guess, Blue, 789, 45.33)). When this is the case, you'll see the individual inputs denoted with a subscript, as in X1, X2, X3, and so on.

2.15 The Tools for Machine Learning

Many tools have been developed to make machine learning more powerful and easier to implement. On this page, we'll take a look at the typical components you might employ in a machine learning ecosystem. You don't need to understand the details of these tools at this stage, and we don't assume you've had previous experience with them. Our goal at this point is simply to give you some idea of what some of the popular tools are and how they relate to one another.

https://photos.app.goo.gl/cLuL2YwEw2UbqL8U8





The Machine Learning Ecosystem
A typical machine learning ecosystem is made up of three main components:

1. Libraries. When you're working on a machine learning project, you likely will not want to write all of the necessary code yourself—instead, you'll want to make use of code that has already been created and refined. That's where libraries come in. A library is a collection of pre-written (and compiled) code that you can make use of in your own project. NumPy is an example of a library popularly used in data science, while TensorFlow is a library specifically designed for machine learning. Read this article for some other useful library.

2. Development environments. A development environment is a software application (or sometimes a group of applications) that provide a whole suite of tools designed to help you (as the developer or machine learning engineer) build out your projects. Jupyter Notebooks and Visual Studio are examples of development environments that are popular for coding many different types of projects, including machine learning projects.

3. Cloud services. A cloud service is a service that offers data storage or computing power over the Internet. In the context of machine learning, you can use a cloud service to access a server that is likely far more powerful than your own machine, or that comes equipped with machine learning models that are ready for you to use. Read more information about different cloud services from this article
For each of these components, there are multiple options you can choose from. Let's have a look at some examples.
Notebooks

Notebooks are originally created as a documenting tool that others can use to reproduce experiments. Notebooks typically contain a combination of runnable code, output, formatted text, and visualizations. One of the most popular open-source notebooks used today by data scientists and data science engineers is Jupyter notebook, which can combine code, formatted text (markdown) and visualization.

Notebooks contains several independent cells that allow for the execution of code snippets within those cells. The output of each cell can be saved in the notebook and viewed by others.

End-to-end with Azure
You can analyze and train a small amount of data with your local machine using Jupyter notebook, Visual studio, or other tools. But with very large amounts of data, or you need a faster processor, it's a better idea to train and test the model remotely using cloud services such as Microsoft Azure. You can use Azure Data Science Virtual Machine, Azure Databricks, Azure Machine Learning Compute, or SQL server ML services to train and test models and use Azure Kubernetes to deploy models.
 

2.16 Libraries for Machine Learning

https://photos.app.goo.gl/PoPtr9oPQdJttHb76




For your reference, here are all the libraries we went over in the video. This is a lot of info; you should not feel like you need to be deeply knowledgeable about every detail of these libraries. Rather, we suggest that you become familiar with what each library is for, in general terms. For example, if you hear someone talking about matplotlib, it would be good for you to recognize that this is a popular library for data visualization. Or if you see a reference to TensorFlow, it would be good to recognize this as a popular machine learning library.

Core Framework and Tools
•	Python is a very popular high-level programming language that is great for data science. Its ease of use and wide support within popular machine learning platforms, coupled with a large catalog of ML libraries, has made it a leader in this space.
•	Pandas is an open-source Python library designed for analyzing and manipulating data. It is particularly good for working with tabular data and time-series data.
•	NumPy, like Pandas, is a Python library. NumPy provides support for large, multi-dimensional arrays of data, and has many high-level mathematical functions that can be used to perform operations on these arrays.
Machine Learning and Deep Learning
•	Scikit-Learn is a Python library designed specifically for machine learning. It is designed to be integrated with other scientific and data-analysis libraries, such as NumPy, SciPy, and matplotlib (described below).
•	Apache Spark is an open-source analytics engine that is designed for cluster-computing and that is often used for large-scale data processing and big data.
•	TensorFlow is a free, open-source software library for machine learning built by Google Brain.
•	Keras is a Python deep-learning library. It provide an Application Programming Interface (API) that can be used to interface with other libraries, such as TensorFlow, in order to program neural networks. Keras is designed for rapid development and experimentation.
•	PyTorch is an open source library for machine learning, developed in large part by Facebook's AI Research lab. It is known for being comparatively easy to use, especially for developers already familiar with Python and a Pythonic code style.
Data Visualization
•	Plotly is not itself a library, but rather a company that provides a number of different front-end tools for machine learning and data science—including an open source graphing library for Python.
•	Matplotlib is a Python library designed for plotting 2D visualizations. It can be used to produce graphs and other figures that are high quality and usable in professional publications. You'll see that the Matplotlib library is used by a number of other libraries and tools, such as SciKit Learn (above) and Seaborn (below). You can easily import Matplotlib for use in a Python script or to create visualizations within a Jupyter Notebook.
•	Seaborn is a Python library designed specifically for data visualization. It is based on matplotlib, but provides a more high-level interface and has additional features for making visualizations more attractive and informative.
•	Bokeh is an interactive data visualization library. In contrast to a library like matplotlib that generates a static image as its output, Bokeh generates visualizations in HTML and JavaScript. This allows for web-based visualizations that can have interactive features.
 

2.17 Cloud Services for Machine Learning

https://photos.app.goo.gl/dtbM4Ni9G7bgEhmZA

A typical cloud service for machine learning provides support for managing the core assets involved in machine learning projects. For your reference, you can see a table summarizing these main assets below. We'll explore all of these components in more detail as we go through the course.

Feature	Description
Datasets	Define, version, and monitor datasets used in machine learning runs.
Experiments / Runs	Organize machine learning workloads and keep track of each task executed through the service.
Pipelines	Structured flows of tasks to model complex machine learning flows.
Models	Model registry with support for versioning and deployment to production.
Endpoints	Expose real-time endpoints for scoring as well as pipelines for advanced automation.

Machine learning cloud services also need to provide support for managing the resources required for running machine learning tasks:

Feature	Description
Compute	Manage compute resources used by machine learning tasks.
Environments	Templates for standardized environments used to create compute resources.
Datastores	Data sources connected to the service environment (e.g. blob stores, file shares, Data Lake stores, databases).






A Brief Intro to Azure Machine Learning

Below are some of the features of Azure Machine Learning that we just discussed. We'll get some hands-on experience using these features during the labs found throughout this course. For now, our goal is just to take a brief tour of the main features.

https://photos.app.goo.gl/2uVCZ8DHF5GjdCH8A

Following are some of the features in Azure ML workspace, a centralized place to work with all the artifacts you create:

Feature	Description
Automated ML	Automate intensive tasks that rapidly iterate over many combinations of algorithms, hyperparameters to find the best model based on the chosen metric.
Designer	A drag-and-drop tool that lets you create ML models without a single line of code.
Datasets	A place you can create datasets.
Experiments	A place that helps you organize your runs.
Models	A place to save all the models created in Azure ML or trained outside of Azure ML.
Endpoints	A place stores real-time endpoints for scoring and pipeline endpoints for advanced automation.
Compute	A designated compute resource where you run the training script or host the service deployment.
Datastores	An attached storage account in which you can store datasets.
 

2.18 Models vs. Algorithms

https://photos.app.goo.gl/DQLg4zJ6PvNGUa7u5



In machine learning, the key distinction of a model and an algorithm is:
Models are the specific representations learned from data
Algorithms are the processes of learning

We can think of the algorithm as a function—we give the algorithm data and it produces a model:
Model = Algorithm(Data)Model=Algorithm(Data)
On the next page, we'll look at this distinction in the context of a concrete example: Linear regression.

More About Machine Learning Algorithms
We can think of an algorithm as a mathematical tool that can usually be represented by an equation as well as implemented in code. For example, y = Wx + b is an algorithm that can be used to calculate y from x if the values for W and b are known. But how do we get W and b?
This is the learning part of machine learning; That is, we can learn these values from training data. For example, suppose the following data are collected:
x	y
1	1
2	2
3	3

We can plug the data into the algorithm and calculate W = 1 and b = 0. We would say that that the algorithm was run on the data and learned the values ofr W and b. The output of the learning process is W = 1 and b = 0.

Machine Learning Models
Machine learning models are outputs or specific representations of algorithms that run on data. A model represents what is learned by a machine learning algorithm on the data.
In the previous example, y = 1*x + 0 is the model we obtained from running the algorithm y = Wx + b on the training data. We can also say that y = 1*x + 0 is the model that can be used to predict y from x.
A machine learning model can also be written in a set of weights or coefficients instead of a full equation. Looking at the previous example, since we know the algorithm, it is redundant to keep the full equation y = 1*x + 0. All we need are the weights (or coefficients) W = 1 and b = 0. Thus, we can also think of a model as a set of weights (or coefficients) that have been learned.
 

2.19 Prelaunch Lab

2.20 Linear Regression
In our first lab, we're going to use Azure Machine Learning Studio to train a model using one of the fundamental machine learning algorithms: Linear regression. Before we dive into the lab, let's review what linear regression is and how it can be used to train a model.
The video below gives a brief review of the main concepts. If you've never seen linear regression before, or need a more thorough review, you can continue on for a detailed explanation below.

https://photos.app.goo.gl/3ydmNgDr4LKtT2aZ9
If you feel confident in your understanding of linear regression, feel free to move ahead to the next page and get started on the lab. Otherwise, we'll go over the concepts in further detail below.

Understanding Linear Regression

As the term suggests, linear regression is an algorithm that uses a straight line (or plane) to describe relationships between variables.
Let's consider a very simple example of a linear relationship. Suppose that we want to know if the number of hours a student spends studying for a test is related to the number of questions answered correctly on the test.
Such a relationship might look something like this:

 

We can see that there is a clear relationship: Students who spent more time studying also scored higher on the test.
What is more, we can see that the data points cluster around a straight line. Linear regression is all about finding the line that best fits the data. And this model (or line) can then be used to make predictions. In this case, if we know the number of hours a student has studied for the test, we can predict how many questions he or she can answer correctly. To make this prediction, we need the equation for the line of best fit. What would that look like?

Simple Linear Regression

You may recall from fundamental algebra that the general equation for a line looks like this:
y = mx + b

Where mm is called the slope of the line, and b is the y-intercept. Again, this is the general equation. For a specific line, we need to know the values for the slope and y-intercept. For example, the following equations represent three different lines.

y = 10x + 50
y = 2x + 3
y = −10x + 40

Equations like these can be used to make predictions. Once we know m and b, we can feed in a value for x and the equation will give us the value of y.
 
 

Linear Regression in Machine Learning

The equation we used above was:
y = mx + b

In algebraic terms, we may refer to mm as the coefficient of x or simply the slope of the line, and we may call bb the y-intercept. In machine learning, you will typically see the y-intercept referred to as the bias. In machine learning, you will also often see the equation represented using different variables, as in:

y = B0 + B1 ∗ x

The letters are different and the order has been changed, but it is exactly the same equation. Thus, we can see that what we know from algebra as the basic equation for a line is also, in machine learning, the equation used for simple linear regression.

Multiple Linear Regression
In more complex cases where there is more than one input variable, we might see something like this:

y = B0 + B1 ∗ x1 + B2 ∗ x2 + B3 ∗ x3 ... + Bn ∗ xn

In this case, we are using multiple input variables to predict the output. When we have multiple input variables like this, we call it multiple linear regression. The visualization of multiple linear regression is no longer a simple line, but instead a plane in multiple dimensions:

 
But don't let any of this intimidate you: The core idea is still that we are modeling a relationship (using a line or plane) in order to help us predict the value of some variable that we are interested in.
Training a Linear Regression Model
To "train a linear regression model" simply means to learn the coefficients and bias that best fit the data. This is the purpose of the linear regression algorithm. Here we will give you a high-level introduction so that you understand conceptually how it works, but we will not go into the mathematical details.
The Cost Function
Notice from our example of test scores earlier that the line we came up with did not perfectly fit the data. In fact, most of the data points were not on the line! When we predict that a student who studies for 10 hours will get a score of 153, we do not expect their score to be exactly 153. Put another way, when we make a prediction using the line, we expect the prediction to have some error.
The process of finding the best model is essentially a process of finding the coefficients and bias that minimize this error. To calculate this error, we use a cost function. There are many cost functions you can choose from to train a model and the resulting error will be different depending one which cost function you choose. The most commonly used cost function for linear regression is the root mean squared error (RMSE)
Preparing the Data
There are several assumptions or conditions you need to keep in mind when you use the linear regression algorithm. If the raw data does not meet these assumptions, then it needs to be prepared and transformed prior to use.
•	Linear assumption: As we've said earlier, linear regression describes variables using a line. So the relationship between the input variables and the output variable needs to be a linear relationship. If the raw data does not follow a linear relationship, you may be able to transform) your data prior to using it with the linear regression algorithm. For example, if your data has an exponential relationship, you can use log transformation.
•	Remove collinearity: When two variables are collinear, this means they can be modeled by the same line or are at least highly correlated; in other words, one input variable can be accurately predicted by the other. For example, suppose we want to predict education level using the input variables number of years studying at school, if an individual is male, and if an individual is female. In this case, we will see collinearity—the input variable if an individual is female can be perfectly predicted by if an individual is male, thus, we can say they are highly correlated. Having highly correlated input variables will make the model less consistent, so it's important to perform a correlation check among input variables and remove highly correlated input variables.
•	Gaussian (normal) distribution: Linear regression assumes that the distance between output variables and real data (called residual) is normally distributed. If this is not the case in the raw data, you will need to first transform the data so that the residual has a normal distribution.
•	Rescale data: Linear regression is very sensitive to the distance among data points, so it's always a good idea to normalize or standardize the data.
•	Remove noise: Linear regression is very sensitive to noise and outliers in the data. Outliers will significantly change the line learned, as shown in the picture below. Thus, cleaning the data is a critical step prior to applying linear regression.

 
Calculating the Coefficients
We've discussed here the overall concept of training a linear regression model: We take the general equation for a line and use some data to learn the coefficients for a specific line that will best fit the data. Just so that you have an idea of what this looks like in concrete terms, let's look at the formulas used to calculate the coefficients. We're showing these in order to give you a general idea of what the calculations actually involve on a concrete level. For this course, you do not need to worry about how the formulas are derived and how to use them to calculate the coefficients.

The formula for getting the slope of the line looks something like this:

 

To get the intercept, we calculate:

 

And to get the root mean squared error (RMSE), we have:

 

In most machine learning libraries (such as Sklearn or Pythorch) the inner workings of the linear regression algorithm are implemented for you. The error and the best coefficients will be automatically calculated when you input the data. Here, the important thing is to understand what is happening conceptually—namely, that we choose a cost function (like RMSE) to calculate the error and then minimize that error in order to arrive at a line of best fit that models the training data and can be used to make predictions.

Now that we've review the concept, let's get some hands-on practice implementing the linear regression algorithm in Azure Machine Learning Studio!

2.21 Linear Regression: Check Your Understanding

Before going on to the lab, here are some additional practice questions that you can use to check your understanding of linear regression.
    

2.22 Lab Instructions

IMPORTANT: Please review the following before going into the lab.
1.The best experience is observed to be on Chrome browser
2.Enable cookies if you haven't already. If you do not you will see an error that looks like the image below - enable your cookies and try again
 
3.After a lab is loaded into your workspace, you have a specific amount of time to finish the lab, it is noted on the right top part of the getting started guide, such as the image below. If you run out of time, you will get a chance to extend it further by 30 mins after which the lab will be deleted.
 

4.If your lab is opened with no activity (idle) for 10 mins., you will see a popup like the following, which you need to respond within 2 mins, otherwise, the lab will be deleted.
 
 
5.If it is hard for you to work on your lab with the guide on the right panel, you can hide that panel so the lab extends to the full workspace you are in. You can always click the right panel arrow buttons to expand the lab guide when and if needed.
 


 

6.If you still need help, please capture the following and email to Udacity Support
Subject Line: Error in Lab name
Timestamp and Timezone:
Brief description of what they were doing: (just clicked the button or doing something else)
Screenshots:




2.23 Lab: Train a Linear Regression Model

Lab Overview
Azure Machine Learning designer (preview) gives you a cloud-based interactive, visual workspace that you can use to easily and quickly prep data, train and deploy machine learning models. It supports Azure Machine Learning compute, GPU or CPU. Machine Learning designer also supports publishing models as web services on Azure Kubernetes Service that can easily be consumed by other applications.
In this lab, we will be using a subset of NYC Taxi & Limousine Commission - green taxi trip records available from Azure Open Datasets. The data is enriched with holiday and weather data. Based on the enriched dataset, we will learn to use the Azure Machine Learning Graphical Interface to process data, build, train, score, and evaluate a regression model to predict NYC taxi fares. To train the model, we will create Azure Machine Learning Compute resource. We will do all of this from the Azure Machine Learning designer without writing a single line of code.
Exercise 1: Register Dataset with Azure Machine Learning studio
Task 1: Upload Dataset
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	From the studio, select Datasets, + Create dataset, From web files. This will open the Create dataset from web files dialog on the right.
 
5.	In the Web URL field provide the following URL for the training data file:
https://introtomlsampledata.blob.core.windows.net/data/nyc-taxi/nyc-taxi-sample-data.csv
6.	Provide nyc-taxi-sample-data as the Name, leave the remaining values at their defaults and select Next.
 
Task 2: Preview Dataset
1.	On the Settings and preview panel, set the column headers drop down to All files have same headers.
2.	Scroll the data preview to right to observe the target column: totalAmount. After you are done reviewing the data, select Next
 
Task 3: Select Columns
1.	Select columns from the dataset to include as part of your training data. Leave the default selections and select Next
 
Task 4: Create Dataset
1.	Confirm the dataset details and select Create
 
Exercise 2: Create New Training Pipeline
Task 1: Open Pipeline Authoring Editor
1.	From the studio, select Designer, +. This will open a visual pipeline authoring editor.
 
Task 2: Setup Compute Target
1.	In the settings panel on the right, select Select compute target.
 
2.	In the Set up compute target editor, select the available compute, and then select Save.
Note: If you are facing difficulties in accessing pop-up windows or buttons in the user interface, please refer to the Help section in the lab environment.
 
Task 3: Add Dataset
1.	Select Datasets section in the left navigation. Next, select My Datasets, nyc-taxi-sample-data and drag and drop the selected dataset on to the canvas.
 
Task 4: Split Dataset
1.	Select Data Transformation section in the left navigation. Follow the steps outlined below:
1.	Select the Split Data prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Fraction of rows in the first output dataset: 0.7
4.	Connect the Dataset to the Split Data module
 
Note that you can submit the pipeline at any point to peek at the outputs and activities. Running pipeline also generates metadata that is available for downstream activities such selecting column names from a list in selection dialogs.
Task 5: Initialize Regression Model
1.	Select Machine Learning Algorithms section in the left navigation. Follow the steps outlined below:
1.	Select the Linear Regression prebuilt module
2.	Drag and drop the selected module on to the canvas
 
Task 6: Setup Train Model Module
1.	Select Model Training section in the left navigation. Follow the steps outlined below:
1.	Select the Train Model prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the Linear Regression module to the first input of the Train Model module
4.	Connect the first output of the Split Data module to the second input of the Train Model module
5.	Select the Edit column link to open the Label column editor
 
2.	The Label column editor allows you to specify your Label or Target column. Type in the label column name totalAmount and then select Save.
 
Task 7: Setup Score Model Module
1.	Select Model Scoring & Evaluation section in the left navigation. Follow the steps outlined below:
1.	Select the Score Model prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the Train Model module to the first input of the Score Model module
4.	Connect the second output of the Split Data module to the second input of the Score Model module
 
Note that Split Data module will feed data for both model training and model scoring. The first output (0.7 fraction) will connect with the Train Model module and the second output (0.3 fraction) will connect with the Score Model module.
Task 8: Setup Evaluate Model Module
1.	Select Model Scoring & Evaluation section in the left navigation. Follow the steps outlined below:
1.	Select the Evaluate Model prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the Score Model module to the first input of the Evaluate Model module
 
Exercise 3: Submit Training Pipeline
Task 1: Create Experiment and Submit Pipeline
1.	Select Submit to open the Setup pipeline run editor.
 
Please note that the button name in the UI is changed from Run to Submit.
2.	In the Setup pipeline run editor, select Experiment, Create new and provide New experiment name: designer-run, and then select Submit.
 
3.	Wait for pipeline run to complete. It will take around 8 minutes to complete the run.
4.	While you wait for the model training to complete, you can learn more about the training algorithm used in this lab by selecting Linear Regression module.
Exercise 4: Visualize Training Results
Task 1: Visualize the Model Predictions
1.	Select Score Model, Outputs, Visualize to open the Score Model result visualization dialog.
 
2.	Observe the predicted values under the column Scored Labels. You can compare the predicted values (Scored Labels) with actual values (totalAmount).
 
Task 2: Visualize the Evaluation Results
1.	Select Evaluate Model, Outputs, Visualize to open the Evaluate Model result visualization dialog.
 
2.	Evaluate the model performance by reviewing the various evaluation metrics, such as Mean Absolute Error, Root Mean Squared Error, etc.
 
Next Steps
Congratulations! You have trained and evaluated your first machine learning model. You can continue to experiment in the environment but are free to close the lab environment tab and return to the Udacity portal to continue with the lesson.
2.24 Walkthrough: Train a Linear Regression Model

In this next video, we'll walk through the lab you just did, highlighting the key steps and concepts. You'll find a walkthrough video like this one after every lab in this course—so if you ever get stuck on a lab, be sure to check out the corresponding walkthrough.

https://photos.app.goo.gl/YYDg7RZAAd9LW6Uz8

2.25 Learning Functions

As mentioned earlier, we can generally think of a machine learning algorithm as a process for learning, and models as specific representations that we train using data. In essence, machine learning algorithms aim to learn a target function (f) that describes the mapping between data input variables (X) and an output variable (Y).

Y = f(X)

The core goal is to learn a useful transformation of the input data that gets us closer to the expected output.
Since the process extrapolates from a limited set of values, there will always be an error e which is independent of the input data (X) such that:

Y = f(X) + e

The variable ee is called irreducible error because no matter how good we get at estimating the target function (f), we cannot reduce this error.

https://photos.app.goo.gl/AvwJW8q6CjbaQeb98

Note that the irreducible error we're discussing here is different from the model error we talked about earlier in the lesson. Irreducible error is caused by the data collection process—such as when we don't have enough data or don't have enough data features. In contrast, the model error measures how much the prediction made by the model is different from the true output. The model error is generated from the model and can be reduced during the model learning process.
 
 

2.26 Parametric vs. Non-parametric

Based on the assumptions about the shape and structure of the function they try to learn, machine learning algorithms can be divided into two categories: parametric and nonparametric.

https://photos.app.goo.gl/oGT4JBXBNSvHY6sZ9

Parametric machine learning algorithms make assumptions about the mapping function and have a fixed number of parameters. No matter how much data is used to learn the model, this will not change how many parameters the algorithm has. With a parametric algorithm, we are selecting the form of the function and then learning its coefficients using the training data.

An example of this would be the approach used in linear regression algorithms, where the simplified functional form can be something like:

B0 + B1 ∗ X1 + B2 ∗ X2 = 0

This assumption greatly simplifies the learning process; after selecting the initial function, the remaining problem is simply to estimate the coefficients B0, B1, and B2 using different samples of input variables X1 and X2.


Benefits:

•	Simpler and easier to understand; easier to interpret the results
•	Faster when talking about learning from data
•	Less training data required to learn the mapping function, working well even if the fit to data is not perfect

Limitations:

•	Highly constrained to the specified form of the simplified function
•	Limited complexity of the problems they are suitable for
•	Poor fit in practice, unlikely to match the underlying mapping function.

Non-parametric Machine Learning Algorithms

Non-parametric algorithms do not make assumptions regarding the form of the mapping function between input data and output. Consequently, they are free to learn any functional form from the training data.

A simple example is the K-nearest neighbors (KNN) algorithm, which we'll discuss in more detail later in the course. KNN does not make any assumptions about the functional form, but instead uses the pattern that points have similar output when they are close.

Benefits:

•	High flexibility, in the sense that they are capable of fitting a large number of functional forms
•	Power by making weak or no assumptions on the underlying function
•	High performance in the prediction models that are produced

Limitations:

•	More training data is required to estimate the mapping function
•	Slower to train, generally having far more parameters to train
•	Overfitting the training data is a risk; overfitting makes it harder to explain the resulting predictions

 

 
 

2.27 Classical ML vs. Deep Learning

https://photos.app.goo.gl/zjRrXzb3TeUAJ1LG6

Remember, all deep learning algorithms are machine learning algorithms but not all machine learning algorithms are deep learning algorithms.

Deep learning algorithms are based on neural networks and the classical ML algorithms are based on classical mathematical algorithms, such as linear regression, logistic regression, decision tree, SVM, and so on.

Deep learning advantages:

•	Suitable for high complexity problems
•	Better accuracy, compared to classical ML
•	Better support for big data
•	Complex features can be learned


Deep learning disadvantages:

•	Difficult to explain trained data
•	Require significant computational power

Classical ML advantages:

•	More suitable for small data
•	Easier to interpret outcomes
•	Cheaper to perform
•	Can run on low-end machines
•	Does not require large computational power

Classical ML disadvantages:

•	Difficult to learn large datasets
•	Require feature engineering
•	Difficult to learn complex functions

 


 

2.28 Approaches to Machine Learning

There are three main approaches to machine learning:
•	Supervised learning
•	Unsupervised learning
•	Reinforcement learning

We'll take a short, high-level look at these approaches here, and then revisit them in more detail in later lessons.

https://photos.app.goo.gl/D3BQkX7y2dT1YCvT7
Supervised learning

Learns from data that contains both the inputs and expected outputs (e.g., labeled data). Common types are:

•	Classification: Outputs are categorical.
•	Regression: Outputs are continuous and numerical.
•	Similarity learning: Learns from examples using a similarity function that measures how similar two objects are.
•	Feature learning: Learns to automatically discover the representations or features from raw data.
•	Anomaly detection: A special form of classification, which learns from data labeled as normal/abnormal.

Unsupervised learning

Learns from input data only; finds hidden structure in input data.

•	Clustering: Assigns entities to clusters or groups.
•	Feature learning: Features are learned from unlabeled data.
•	Anomaly detection: Learns from unlabeled data, using the assumption that the majority of entities are normal.

Reinforcement learning

Learns how an agent should take action in an environment in order to maximize a reward function.

•	Markov decision process: A mathematical process to model decision-making in situations where outcomes are partly random and partly under the control of a decision-maker. Does not assume knowledge of an exact mathematical model.

The main difference between reinforcement learning and other machine learning approaches is that reinforcement learning is an active process where the actions of the agent influence the data observed in the future, hence influencing its own potential future states. In contrast, supervised and unsupervised learning approaches are passive processes where learning is performed without any actions that could influence the data.

 
 

 

 
2.29 The Trade-Offs
As all things in computer science, machine learning involves certain trade-offs. Two of the most important are bias vs. variance and overfitting vs. underfitting.

https://photos.app.goo.gl/3cGt3mVCwYVa4Rxo9

Bias vs. Variance
Bias measures how inaccurate the model prediction is in comparison with the true output. It is due to erroneous assumptions made in the machine learning process to simplify the model and make the target function easier to learn. High model complexity tends to have a low bias.
Variance measures how much the target function will change if different training data is used. Variance can be caused by modeling the random noise in the training data. High model complexity tends to have a high variance.
As a general trend, parametric and linear algorithms often have high bias and low variance, whereas non-parametric and non-linear algorithms often have low bias and high variance

Overfitting vs. Underfitting
Overfitting refers to the situation in which models fit the training data very well, but fail to generalize to new data.
Underfitting refers to the situation in which models neither fit the training data nor generalize to new data.
 

https://photos.app.goo.gl/YSqakmwtM21zZS4V9





Bias vs. Variance Trade-off

The prediction error can be viewed as the sum of model error (error coming from the model) and the irreducible error (coming from data collection).

prediction error = Bias error + variance + error + irreducible error

Low bias means fewer assumptions about the target function. Some examples of algorithms with low bias are KNN and decision trees. Having fewer assumptions can help generalize relevant relations between features and target outputs. In contrast, high bias means more assumptions about the target function. Linear regression would be a good example (e.g., it assumes a linear relationship). Having more assumptions can potentially miss important relations between features and outputs and cause underfitting.

Low variance indicates changes in training data would result in similar target functions. For example, linear regression usually has a low variance. High variance indicates changes in training data would result in very different target functions. For example, support vector machines usually have a high variance. High variance suggests that the algorithm learns the random noise instead of the output and causes overfitting.
Generally, increasing model complexity would decrease bias error since the model has more capacity to learn from the training data. But the variance error would increase if the model complexity increases, as the model may begin to learn from noise in the training data.

The goal of training machine learning models is to achieve low bias and low variance. The optimal model complexity is where bias error crosses with variance error.

https://photos.app.goo.gl/H5g1YS1E8Ptvq1kF8

Overfitting vs. Underfitting

•	k-fold cross-validation: it split the initial training data into k subsets and train the model k times. In each training, it uses one subset as the testing data and the rest as training data.
•	hold back a validation dataset from the initial training data to estimatete how well the model generalizes on new data.
•	simplify the model. For example, using fewer layers or less neurons to make the neural network smaller.
•	use more data.
•	reduce dimensionality in training data such as PCA: it projects training data into a smaller dimension to decrease the model complexity.
•	Stop the training early when the performance on the testing dataset has not improved after a number of training iterations.
 
 
 

2.30 Lesson Summary

https://photos.app.goo.gl/UkQzrozNp3Y5ZmJG9

In this lesson, our goal was to give you a high-level introduction to the field of machine learning, including the broader context in which this branch of computer science exists.

Here are the main topics we covered:

•	What machine learning is and why it's so important in today's world
•	The historical context of machine learning
•	The data science process
•	The types of data that machine learning deals with
•	The two main perspectives in ML: the statistical perspective and the computer science perspective
•	The essential tools needed for designing and training machine learning models
•	The basics of Azure ML
•	The distinction between models and algorithms
•	The basics of a linear regression model
•	The distinction between parametric vs. non-parametric functions
•	The distinction between classical machine learning vs. deep learning
•	The main approaches to machine learning
•	The trade-offs that come up when making decisions about how to design and training machine learning models

In the process, you also trained your first machine learning model using Azure Machine Learning Studio.

3 Model Training

3.1 Lesson Overview

https://photos.app.goo.gl/qdb21Kvf6kdxo6oVA

Before training a model, we first need to handle data preparation, so we'll explore this topic first. More specifically, we will go over:

•	Data importing and transformation
•	The data management process, including:

•	The use of datastores and datasets
•	Versioning
•	Feature engineering
•	How to monitor for data drift

Next, we will introduce the basics of model training. We'll cover:

•	The core model training process
•	Two of the fundamental machine learning models: Classifier and regressor
•	
•	The model evaluation process and relevant metrics
And finally, we'll conclude with an introduction to ensemble learning and automated machine learning, two core techniques used to make decisions based on multiple—rather than single—trained models.

3.2 Prelaunch Lab

3.3 Lab Instructions

Please review the following before going into the lab.
1.	The best experience is observed to be on Chrome browser
2.	Enable cookies if you haven't already. If you do not, you will see an error that looks like the image below - enable your cookies and try again
 
3.After a lab is loaded into your workspace, you have specific amount of time to finish the lab, it is noted on the right top part of the getting started guide, such as the image below. If you run out of time, you will get a chance to extend it further by 30 mins after which the lab will be deleted.

 

4.If your lab is opened with no activity (idle) for 10 mins., you will see a popup like the following, which you need to respond within 2 mins, otherwise, the lab will be deleted. The 10 mins., Idle timeout is especially important for folks who are opening up the lab in a different tab.
 
 
5.If it is hard for you to work on your lab with the guide on the right panel, you can hide that panel so the lab extends to the full workspace you are in. You can always click the right panel arrow buttons to expand the lab guide when and if needed.

 


 


6.If you still need help, please capture the following and email to Udacity Support
Subject Line: Error in Lab name
Timestamp and Timezone:
Brief description of what they were doing: (just clicked the button or doing something else)
Screenshots:

3.4 Data Import and Transformation

https://photos.app.goo.gl/ru96zw5QL1zQn5G36

Data wrangling is the process of cleaning and transforming data to make it more appropriate for data analysis. The process generally follows these main steps:

•	Explore the raw data and check the general quality of the dataset.
•	Transform the raw data, by restructuring, normalizing, and cleaning the data. For example, this could involve handling missing values and detecting errors.
•	Validate and publish the data.

Data wrangling is an iterative process where you do some data transformation then check the results and come back to the process to make improvements.
 

The best way to understanding data wrangling is to get hands-on experience actually doing it—so that's what we'll do in our next lab.

3.5 Lab: Import, Transform, and Export Data

Import, transform and export data
Lab Overview
In this lab you learn how to import your own data in the designer to create custom solutions. There are two ways you can import data into the designer in Azure Machine Learning Studio:
•	Azure Machine Learning datasets
Register datasets in Azure Machine Learning to enable advanced features that help you manage your data.
•	Import Data module
Use the Import Data module to directly access data from online datasources.
The first approach will be covered later in the next lab, which focuses on registering and versioning a dataset in Azure Machine Learning studio.
While the use of datasets is recommended to import data, you can also use the Import Data module from the designer. Data comes into the designer from either a Datastore or from Tabular Datasets. Datastores will be covered later in this course, but just for a quick definition, you can use Datastores to access your storage without having to hard code connection information in your scripts. As for the second option, the Tabular datasets, the following datasources are supported in the designer: Delimited files, JSON files, Parquet files or SQL queries.
The following exercise focuses on the Import Data module to load data into a machine learning pipeline from several datasets that will be merged and restructured. We will be using some sample data from the UCI dataset repository to demonstrate how you can perform basic data import transformation steps with the modules available in Azure Machine Learning designer.
Exercise 1: Import, transform and export data using the Visual Pipeline Authoring Editor
Task 1: Open Pipeline Authoring Editor
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	From the studio, select Designer, +. This will open a visual pipeline authoring editor.
 
Task 2: Setup Compute Target
1.	In the settings panel on the right, select Select compute target.
 
2.	In the Set up compute target editor, select the existing compute target, and then select Save.
Note: If you are facing difficulties in accessing pop-up windows or buttons in the user interface, please refer to the Help section in the lab environment.
 
Task 3: Import data from Web URL
1.	Select Data Input and Output section in the left navigation. Next, select Import Data and drag and drop the selected module on to the canvas.
 
2.	In the Import data panel on the right, select the URL via HTTP option in the Data Source drop-down and provide the following Data source URL for the first CSV file you will import in your pipeline: https://introtomlsampledata.blob.core.windows.net/data/crime-data/crime-dirty.csv
 
3.	Select the Preview schema to filter the columns you want to include. You can also define advanced settings like Delimiter in Parsing options. Select Save to close the dialog.
 
Task 4: Create Experiment and Submit Pipeline
1.	Back to the pipeline canvas, select Submit on the top right corner to open the Setup pipeline run editor.
2.	In the Setup pipeline run editor, select Experiment, Create new and provide New experiment name: designer-data-import, and then select Submit.
 
Please note that the button name in the UI is changed from Run to Submit.
3.	Wait for pipeline run to complete. It will take around 10 minutes to complete the run.
Task 5: Visualize Import Data results
1.	Select the Import Data module on the canvas and then select Outputs on the right pane. Click on the Visualize icon to open the Import Data result visualization dialog.
 
2.	In the Import Data result visualization dialog take some moments to explore all the metadata that is now available to you, such as: number of rows, columns, preview of data and for each column you select you can observe: Mean, Median, Min, Max and also number of Unique Values and Missing Values. Data profiles help you glimpse into the column types and summary statistics of a dataset. Scroll right and select the X Coordinate column. Notice the Nan value on the third row in the preview table and check the Missing values number in the Statistics section.
 
3.	Select Close to return to the pipeline designer canvas where you can continue the data import phase.
Exercise 2: Restructure the data split across multiple files
Task 1: Append rows from two additional data sources
1.	Select Data Input and Output section in the left navigation. Next, drag and drop two Import Data modules on to the canvas as demonstrated in the first exercise and fill in the Web URLs as follows:
o	for the first one, Data source URL : https://introtomlsampledata.blob.core.windows.net/data/crime-data/crime-spring.csv
o	for the second one, Data source URL : https://introtomlsampledata.blob.core.windows.net/data/crime-data/crime-winter.csv
 
2.	For each of the three Import Data modules, select Preview schema and ensure that the data type for FBI Code and Location is of type String and then select Save.
 
3.	Select the Data Transformation section in the left navigation. Drag and drop the Add rows module and connect it to the above added Import data modules.
 
4.	Repeat the same step and add a second Add rows module that connects the output from the first Import data module to the output of the first Add rows module.
 
Task 2: Clean missing values
1.	Drag the Clean Missing Data module from the Data Transformation section in the left navigation.
 
2.	Select Edit column in the right pane to configure the list of columns to be cleaned. Select Column names from the available include options and type the name of the columns you intend to clean at this step: X Coordinate and Y Coordinate. Select Save to close the dialog.
 
3.	Set the Minimum missing value ratio to 0.1 and the Maximum missing value ratio to 0.5. Select Replace with mean in the Cleaning mode field.
 
Task 3: Submit Pipeline
1.	Select Submit to open the Setup pipeline run editor.
2.	In the Setup pipeline run editor, select Select existing, designer-data-import for Experiment, and then select Submit.
 
Please note that the button name in the UI is changed from Run to Submit.
3.	Wait for pipeline run to complete. It will take around 8 minutes to complete the run.
Task 4: Save the clean dataset
1.	Select the Clean missing data module you created on the canvas and then select Outputs + logs on the right pane. Click on the Save icon under the Cleaned dataset section to open the Save as dataset dialog.
 
2.	Check the option to create a new dataset and enter crime-all in the dataset name field. Select Save to close the dialog.
  
3.	From the left navigation, select Datasets. This will open the Registered datasets page. See your registered dataset among the other datasets you used during this lesson.
 
Next Steps
Congratulations! You completed a few basic steps involved in the data explore and transform process, using the prebuilt modules you can find in the visual editor provided by Azure Machine Learning Studio. You can continue to experiment in the environment but are free to close the lab environment tab and return to the Udacity portal to continue with the lesson.
3.6 Walkthrough: Import, Transform, and Export Data

Lab: Import, Transform, and Export Data

https://photos.app.goo.gl/9w7c66FL9HgZ6rb39

3.7 Managing Data

https://photos.app.goo.gl/zNrWSteD7CiiVQ4E6

As we just discussed, Azure Machine Learning has two data management tools that we need to consider: Datastores and datasets. 

At first the distinction between the two may not be entirely clear, so let's have a closer look at what each one does and how they are related.

Datastores vs. Datasets

https://photos.app.goo.gl/wknztevJiXLz31Kt9

Datastores offer a layer of abstraction over the supported Azure storage services. They store all the information needed to connect to a particular storage service. Datastores provide an access mechanism that is independent of the computer resource that is used to drive a machine learning process.

Datasets are resources for exploring, transforming, and managing data in Azure ML. A dataset is essentially a reference that points to the data in storage. It is used to get specific data files in the datastores.

 
The Data Access Workflow

https://photos.app.goo.gl/L7xJxeEPFUEsBFDW6

The steps of the data access workflow are:

1.	Create a datastore so that you can access storage services in Azure.
2.	
3.	Create a dataset, which you will subsequently use for model training in your machine learning experiment.
4.	
5.	Create a dataset monitor to detect issues in the data, such as data drift.
In the video, we mentioned the concept of data drift. Over time, the input data that you are feeding into your model is likely to change—and this is what we mean by data drift. Data drift can be problematic for model accuracy. Since you trained the model on a certain set of data, it can become increasingly inaccurate and the data changes more and more over time. For example, if you train a model to detect spam in email, it may become less accurate as new types of spam arise that are different from the spam on which the model was trained.
As we noted in the video, you can set up dataset monitors to detect data drift and other issues in your data. When data drift is detected, you can have the system automatically update the input dataset so that you can retrain the model and maintain its accuracy.

 

3.8 Prelaunch Lab

3.9 More About Datasets
On the last page, we discussed the main features of datastores and datasets. Now, let's look a little more closely at how datasets work in Azure Machine Learning.

https://photos.app.goo.gl/xtLU6hC5FZkja7Xu6

Key points to remember about datasets:

•	They are used to interact with your data in the datastore and to package data into consumable objects.
•	They can be created from local files, public URLs, Azure Open Datasets, and files uploaded to the datastores.
•	They are not copies of the data but references that point to the original data. This means that no extra storage cost is incurred when you create a new dataset.
•	Once a dataset is registered in Azure ML workspace, you can share it and reuse it across various other experiments without data ingestion complexities.

https://photos.app.goo.gl/rwsLDVxJbL2jqQyKA

In summary, here are some of the main things that datasets allow you to do:

•	Have a single copy of some data in your storage, but reference it multiple times—so that you don't need to create multiple copies each time you need that data available.
•	Access data during model training without specifying connection strings or data paths.
•	More easily share data and collaborate with other users.
•	Bookmark the state of your data by using dataset versioning

You would do versioning most typically when:
•	New data is available for retraining.
•	When you are applying different approaches to data preparation or feature engineering.

Keep in mind that there are two dataset types supported in Azure ML Workspace:

•	The Tabular Dataset, which represents data in a tabular format created by parsing the provided file or list of files.
•	The Web URL (File Dataset), which references single or multiple files in datastores or from public URLs.

We'll get practice working with these dataset types in the upcoming labs.
 


3.10 Lab: Create and Version a Dataset
Create and version a dataset
Lab Overview
To access your data in your storage account, Azure Machine Learning offers datastores and datasets. Create an Azure Machine Learning datasets to interact with data in your datastores and package your data into a consumable object for machine learning tasks. Register the dataset to your workspace to share and reuse it across different experiments without data ingestion complexities.
Datasets can be created from local files, public urls, Azure Open Datasets, or specific file(s) in your datastores. To create a dataset from an in memory pandas dataframe, write the data to a local file, like a csv, and create your dataset from that file. Datasets aren’t copies of your data, but are references that point to the data in your storage service, so no extra storage cost is incurred.
In this lab, we are using a subset of NYC Taxi & Limousine Commission - green taxi trip records available from Azure Open Datasets to show how you can register and version a Dataset using the AML designer interface. In the first exercises we use a modified version of the original CSV file, which includes collected records for five months (January till May). The second exercise demonstrates how we can create a new version of the initial dataset when new data is collected (in this case, we included records collected in June in the CSV file).
Exercise 1: Register Dataset with Azure Machine Learning studio
Task 1: Upload Dataset from web file
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	From the studio, select Datasets, + Create dataset, From web files. This will open the Create dataset from web files dialog on the right.
 
5.	Provide the following information and then select Next:
1.	Web URL: https://introtomlsampledata.blob.core.windows.net/data/nyc-taxi/nyc-taxi-sample-data-5months.csv
2.	Name: nyc-taxi-sample-dataset
 
Task 2: Preview Dataset
1.	On the Settings and preview panel, set the Column headers drop down to All files have same headers.
2.	Scroll the data preview to right to observe the target column: totalAmount. After you are done reviewing the data, select Next
 
Task 3: Select Columns
1.	Select columns from the dataset to include as part of your training data. Leave the default selections and select Next
 
Task 4: Create Dataset
1.	Confirm the dataset details and select Create
 
Exercise 2: Create a version of the existing Dataset
Task 1: Register new dataset version
1.	From the Azure Machine Learning studio, select Datasets and select the nyc-taxi-sample-dataset dataset created in the first exercise. This will open the Dataset details page.
2.	Select New version, From web files to open the same Create dataset from web files dialog you already entered in the first exercise.
 
3.	This time, the Name and Dataset version fields are already filled in for you. Provide the following information and select Next to move on to the next step:
1.	Web URL: https://introtomlsampledata.blob.core.windows.net/data/nyc-taxi/nyc-taxi-sample-data-6months.csv
 
4.	Select All files have the same headers in the Column headers drop-down and move on to the schema selection step.
5.	On the Schema page, let’s suppose you decided to exclude some columns from your dataset. Exclude columns: snowDepth, prcipTime, precipDepth. Select Next to move on to the final step.
 
6.	Notice the Dataset version value in the basic info section. Select Create to close the new version confirmation page.
 
Task 2: Review both versions of the dataset
1.	Back to the Datasets page, in the Registered datasets list, notice the version value for the nyc-taxi-sample-dataset dataset.
 
2.	Select the nyc-taxi-sample-dataset dataset link to open the dataset details page, where Version 2(latest) is automatically selected. Go to the Explore section to observe the structure and content of the new version. Notice the columns and rows structure in the dataset preview pane:
o	Number of columns: 11
o	Number of rows: 10000
o	Scroll right to check that the three excluded columns are missing (snowDepth, prcipTime, precipDepth)
 
3.	Select Version 1 from the drop-down near the dataset name title and notice the changing values for:
o	Number of columns: 14 (since the previous version still contains the three excluded columns)
o	Number of rows: 9776 (since the previous version contains only data for 5 months)
 
Next Steps
Congratulations! You have now explored a first simple scenario for dataset versioning using the Azure Machine Learning studio. You found out how you can create and version a simple dataset when new training data is available. You can continue to experiment in the environment but are free to close the lab environment tab and return to the Udacity portal to continue with the lesson.
3.11 Walkthrough: Create and Version a Dataset
Lab: Create and Version a Dataset

https://photos.app.goo.gl/55iEQAs4ooN7PPWB9

Keep in mind that datasets and different dataset versions are just references so it is important that the original data files are not changed or overwritten in any way because that will invalidate the datasets and dataset versions.





3.12 Introducing Features
In the previous lesson, we took a look at some examples of tabular data:


SKU	Make	Color	Quantity	Price
908721	Guess	Blue	789	45.33
456552	Tillys	Red	244	22.91
789921	A&F	Green	387	25.92
872266	Guess	Blue	154	17.56
Note: We have been referring to this as a data table, but you will also see data in this format called a matrix. The term matrix is commonly used in mathematics, and it refers to a rectangular array—which, just like a table, contains data arranged in rows and columns.

Recall that the columns in a table can be referred to as features. In the above example, color and quantity are features of the products. In the last lesson, we mentioned briefly that feature engineering is an important part of data preparation. In this lesson, we'll look at this topic in more detail.

https://photos.app.goo.gl/yuZzzoct8fcyChbv8

In many cases, the set of initial features in the data is not enough to produce high quality trained machine learning models. You can use feature engineering to derive new features based on the values of existing features. This process can be as simple as applying a mathematical function to a feature (such as adding 1 to all values in an existing feature ) or it can be as complex as training a separate machine learning model to create values for new features.

Once you have the features, another important task is selecting the features that are most important or most relevant. This process is called feature selection.

Many machine learning algorithms cannot accommodate a large number of features, so it is often necessary to do dimensionality reduction to decrease the number of features.
 

3.13 Feature Engineering

https://photos.app.goo.gl/dveG89NQDLWJc7xh9


 

Examples of Feature Engineering Tasks

https://photos.app.goo.gl/9hhfnKyJ8evgLE1T7


 

Summary of Feature Engineering Approaches by Data Type

https://photos.app.goo.gl/qoM3FE2wsgr2BDRN7


 
3.14 Prelaunch Lab
3.15 Feature Selection
https://photos.app.goo.gl/q84gsYXtFuaajvFY6

There are mainly two reasons for feature selection. Some features might be highly irrelevant or redundant. So it's better to remove these features to simplify the situation and improve performance. Additionally, it may seem like engineering more features is always a good thing, but as we mentioned earlier, many machine learning algorithms suffer from the curse of dimensionality—that is, they do not perform well when given a large number of variables or features.

https://photos.app.goo.gl/SSJmAKDvp6ATLPPN8

We can improve the situation of having too many features through dimensionality reduction.
Commonly used techniques are:
•	PCA (Principal Component Analysis)
•	t-SNE (t-Distributed Stochastic Neighboring Entities)
•	Feature embedding
Azure ML prebuilt modules:
•	Filter-based feature selection: identify columns in the input dataset that have the greatest predictive power
•	Permutation feature importance: determine the best features to use by computing the feature importance scores
 
3.16 Lab: Engineer and Select Features
Engineer and select features
Lab Overview
This lab demonstrates the feature engineering process for building a regression model using bike rental demand prediction as an example. In machine learning predictions, effective feature engineering will lead to a more accurate model. We will use the Bike Rental UCI dataset as the input raw data for this experiment. This dataset is based on real data from the Capital Bikeshare company, which operates a bike rental network in Washington DC in the United States. The dataset contains 17,379 rows and 17 columns, each row representing the number of bike rentals within a specific hour of a day in the years 2011 or 2012. Weather conditions (such as temperature, humidity, and wind speed) were included in this raw feature set, and the dates were categorized as holiday vs. weekday etc.
The field to predict is cnt which contains a count value ranging from 1 to 977, representing the number of bike rentals within a specific hour. Our main goal is to construct effective features in the training data, so we build two models using the same algorithm, but with two different datasets. Using the Split Data module in the visual designer, we split the input data in such a way that the training data contains records for the year 2011, and the testing data, records for 2012. Both datasets have the same raw data at the origin, but we added different additional features to each training set:
•	Set A = weather + holiday + weekday + weekend features for the predicted day
•	Set B = number of bikes that were rented in each of the previous 12 hours
We are building two training datasets by combining the feature set as follows:
•	Training set 1: feature set A only
•	Training set 2: feature sets A+B
For the model, we are using regression because the number of rentals (the label column) contains continuous real numbers. As the algorithm for the experiment, we will be using the Boosted Decision Tree Regression.
Exercise 1: Data pre-processing using the Pipeline Authoring Editor
Task 1: Upload Dataset
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	From the studio, select Datasets, + Create dataset, From web files. This will open the Create dataset from web files dialog on the right.
 
5.	In the Web URL field provide the following URL for the training data file: https://introtomlsampledata.blob.core.windows.net/data/bike-rental/bike-rental-hour.csv
6.	Provide Bike Rental Hourly as the Name, leave the remaining values at their defaults and select Next.
 
7.	Select the option to Use headers from the first file in the Settings and preview dialog and then select Next, Next and Create to confirm all details in registering the dataset.
 
Task 2: Open Pipeline Authoring Editor
1.	From the left navigation, select Designer, +. This will open a visual pipeline authoring editor.
 
Task 3: Setup Compute Target
1.	In the settings panel on the right, select Select compute target.
 
2.	In the Set up compute target editor, select the existing compute target, choose a name for the pipeline draft: Bike Rental Feature Engineering and then select Save.
Note: If you are facing difficulties in accessing pop-up windows or buttons in the user interface, please refer to the Help section in the lab environment.
Task 4: Select columns in the dataset
1.	Drag and drop on the canvas, the available Bike Rental Hourly dataset under the Datasets category on the left navigation.
 
2.	Under the Data transformation category drag and drop the Edit Metadata module, connect the module to the dataset, and select Edit column on the right pane.
 
3.	Add the season and weathersit column and select Save.
 
4.	Configure the Edit metadata module by selecting the Categorical attribute for the two columns.
 
Note that you can submit the pipeline at any point to peek at the outputs and activities. Running pipeline also generates metadata that is available for downstream activities such selecting column names from a list in selection dialogs. Please refer ahead to Exercise 1, Task 8, Step 3 on details of submitting the pipeline. It can take up to 5-10 minutes to run the pipeline.
5.	Under the Data transformation category drag and drop the Select Columns in Dataset module, connect the module to the Edit Metadata module, and select Edit column on the right pane.
 
6.	Configure the Select Columns in Dataset module as follows:
o	Include: All columns
o	Select +
o	Exclude Column names: instant, dteday, casual,registered
o	Select Save
 
Note: You can copy and paste all four column names separated by comma (instant, dteday, casual,registered) in the text box, then select anywhere on the dialog, and then select Save.
7.	Under the Python Language category on the left, select the Execute Python Script module and connect it with the Select Columns in Dataset module. Make sure the connector is connected to the very first input of the Execute Python Script module.
 
8.	We are using the Python script to append a new set of features to the dataset: number of bikes that were rented in each of the previous 12 hours. Feature set B captures very recent demand for the bikes. This will be the B set in the described feature engineering approach.
Select Edit code and use the following lines of code:

# The script MUST contain a function named azureml_main
# which is the entry point for this module.

# imports up here can be used to
import pandas as pd
import numpy as np

# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
def azureml_main(dataframe1 = None, dataframe2 = None):

    # Execution logic goes here
    print(f'Input pandas.DataFrame #1: {dataframe1}')

    # If a zip file is connected to the third input port,
    # it is unzipped under "./Script Bundle". This directory is added
    # to sys.path. Therefore, if your zip file contains a Python file
    # mymodule.py you can import it using:
    # import mymodule

    for i in np.arange(1, 13):
        prev_col_name = 'cnt' if i == 1 else 'Rentals in hour -{}'.format(i-1)
        new_col_name = 'Rentals in hour -{}'.format(i)

        dataframe1[new_col_name] = dataframe1[prev_col_name].shift(1).fillna(0)

    # Return value must be of a sequence of pandas.DataFrame
    # E.g.
    #   -  Single return value: return dataframe1,
    #   -  Two return values: return dataframe1, dataframe2
    return dataframe1,
Don’t worry if you do not fully understand the details of the Python code above. For now, it’s enough to keep in mind that is adds 12 new columns to your dataset containing the number of bikes that were rented in each of the previous 12 hours.
Task 5: Split data into train and test datasets
1.	Use the Split Data module under the Data Transformation module and connect its input with output from the Select Columns in Dataset module. Use the following configuration:
o	Splitting mode: Relative Expression
o	Relational expression: \"yr" == 0
 
2.	Select the Split Data module block and use the menu buttons to Copy and Paste it on the canvas. Connect the second one to the output of the Python Script execution step, which is the featured B set.
 
Task 6: Select columns from the test and training resulted sets
1.	Next, using the Select columns module under the Data transformation category, create four identical modules to exclude the yr column from all the outputs: test and training sets in both branches: A and A+B.
 
2.	Use the following structure for the columns field in each module:
 
Task 7: Create the regression model
1.	Under the Machine Learning Algorithms, Regression category, select the Boosted Decision Tree Regression module. Drag and drop it on the canvas and use the default settings provided.
 
2.	Next, use the Train model module under the Model training category and enter the cnt column in the Label column field.
3.	Link the Boosted Decision Tree Regression module as the first input and the training dataset as the second input like in the image below.
 
4.	Use the exact same configuration on the right branch that uses the output from the Python Script.
 
Task 8: Evaluate and score models
1.	Use two Score Model modules (under the Model Scoring and Evaluation category) and link on the input the two trained models and the test datasets.
2.	Drag the Evaluate Model module which stands in the same category, Model Scoring and Evaluation and link it to the two Score Model modules.
 
3.	Select Submit to open the Setup pipeline run editor. In the Setup pipeline run editor, select Experiment, Create new and provide New experiment name: BikeRentalHourly.
 
Please note that the button name in the UI is changed from Run to Submit.
4.	Wait for pipeline run to complete. It will take around 10 minutes to complete the run.
5.	Once the pipeline execution completes, right click on the Evaluate Model module and select Visualize Evaluation results.
 
6.	The Evaluate Model result visualization popup shows the results of the evaluation.
 
Notice the values for the **Mean_Absolute_Error** metric. The first value (the bigger one) corresponds to the model trained on feature set A. The second value (the smaller one) corresponds to the model trained on feature sets A + B.

It is remarkable how, using simple feature engineering to derive new features from the existing data set, a new context was created that allowed the model to better understand the dynamics of the data and hence, produce a better prediction.

Next Steps
Congratulations! You have trained and compared performance of two models using the same algorithm, but with two different datasets. You can continue to experiment in the environment but are free to close the lab environment tab and return to the Udacity portal to continue with the lesson.
3.17 Walkthrough: Engineer and Select Features
https://photos.app.goo.gl/PC8XBFgw9aZcHnTX7
https://photos.app.goo.gl/154gUDSUgqeJJs8N6

3.18 Data Drift

As we mentioned earlier, data drift is change in the input data for a model. Over time, data drift causes degradation in the model's performance, as the input data drifts farther and farther from the data on which the model was trained.

https://photos.app.goo.gl/iE2KBBmSocSWHvsT7


 

Monitoring for Data Drift

As we noted, data drift is one of the main reasons that model performance gets worse over time. Fortunately, Azure Machine Learning allows you to set up dataset monitors that can alert you about data drift and even take automatic actions to correct data drift.

https://photos.app.goo.gl/XkF5Y45FKPkuuReo9

Remember, the process of monitoring for data drift involves:

•	Specifying a baseline dataset – usually the training dataset
•	Specifying a target dataset – usually the input data for the model
•	Comparing these two datasets over time, to monitor for differences

Here are a couple different types of comparisons you might want to make when monitoring for data drift:

•	Comparing input data vs. training data. This is a proxy for model accuracy; that is, an increased difference between the input vs. training data is likely to result in a decrease in model accuracy.
•	Comparing different samples of time series data. In this case, you are checking for a difference between one time period and another. For example, a model trained on data collected during one season may perform differently when given data from another time of year. Detecting this seasonal drift in the data will alert you to potential issues with your model's accuracy.
 

3.19 Model Training Basics

In this section, we will get into more detail on the steps involved in training a model, and then we'll get some hands-on practice with the process in the upcoming lab.

Remember that our ultimate goal is to produce a model we can use to make predictions. Put another way, we want to be able to give the model a set of input features, X, and have it predict the value of some output feature, y.

https://photos.app.goo.gl/zQkWVJrGFtLc4Gnt7

Parameters and Hyperparameters

When we train a model, a large part of the process involves learning the values of the parameters of the model. For example, earlier we looked at the general form for linear regression:

y = B0 + B1 ∗ x1 + B2 ∗ x2 + B3 ∗ x3 ... + Bn ∗ xn

The coefficients in this equation, B0…Bn, determine the intercept and slope of the regression line. When training a linear regression model, we use the training data to figure out what the value of these parameters should be. Thus, we can say that a major goal of model training is to learn the values of the model parameters.

In contrast, some model parameters are not learned from the data. These are called hyperparameters and their values are set before training. Here are some examples of hyperparameters:

•	The number of layers in a deep neural network
•	The number of clusters (such as in a k-means clustering algorithm)
•	The learning rate of the model

We must choose some values for these hyperparameters, but we do not necessarily know what the best values will be prior to training. Because of this, a common approach is to take a best guess, train the model, and then tune adjust or tune the hyperparameters based on the model's performance.



Splitting the Data

As mentioned in the video, we typically want to split our data into three parts:
•	Training data
•	Validation data
•	Test data

We use the training data to learn the values for the parameters. Then, we check the model's performance on the validation data and tune the hyperparameters until the model performs well with the validation data. For instance, perhaps we need to have more or fewer layers in our neural network. We can adjust this hyperparameter and then test the model on the validation data once again to see if its performance has improved.

Finally, once we believe we have our finished model (with both parameters and hyperparameters optimized), we will want to do a final check of its performance—and we need to do this on some fresh test data that we did not use during the training process.
 

3.20 Model Training in Azure Machine Learning

https://photos.app.goo.gl/LyBn9yAUAYgZ37LdA


3.21 Taxonomy of Azure Machine Learning

https://photos.app.goo.gl/fD1zSy8pHH4DYb5z5

 


3.22 Training Classifiers

As we described in the last lesson, two of the main types of supervised learning are classification and regression. In this section, we'll get some practice training both of these types of models. But first, let's discuss the concepts in more detail—starting with classification.

In a classification problem, the outputs are categorical or discrete.

For example, you might want to classify emails as spam or not spam; each of these is a discrete category.

https://photos.app.goo.gl/GEVhY82JHy6f2mKi9


 

Training Regressors

Now let us turn to regression. The main distinction that sets a regression problem apart from a classification problem is the form of the output:

In a regression problem, the output is numerical or continuous.

A classic example would be a problem in which you are given data concerning houses and then asked to predict the price; this is a regression problem because price is a continuous, numerical output.

https://photos.app.goo.gl/tYbX6kjmRy5xVNhT7

 


3.23 Evaluating Model Performance

It is not enough to simply train a model on some data and then assume that the model will subsequently perform well on future data. Instead, as we've mentioned previously, we need to split off a portion of our labeled data and reserve it for evaluating our model's final performance. We refer to this as the test dataset.

The test dataset is a portion of labeled data that is split off and reserved for model evaluation.

If a model learns to perform well with the training data, but performs poorly with the test data, then there may be a problem that we will need to address before putting our model out into the real world. In practice, we will also need to decide what metrics we will use to evaluate performance, and whether there are any particular thresholds that the model needs to meet on these metrics in order for us to decide that it is "good enough."

https://photos.app.goo.gl/2K4VLKQpUQTT33cS8

When splitting the available data, it is important to preserve the statistical properties of that data. This means that the data in the training, validation, and test datasets need to have similar statistical properties as the original data to prevent bias in the trained model.

 







3.24 Confusion Matrices

Suppose that we have trained a simple binary classification model: Given an image, this model will indicate whether it is a picture of a cat or a picture of a dog. How can we evaluate our model's performance? What is a good metric for doing so?

Let us first consider what it means for the model to perform well. If the model tells us an image has a dog in it and that image actually has a dog, we would say it performs well. And similarly, if it says that the image has a cat and it actually has a cat that would also be good.

To help us think about the problem, we can construct a table that shows all of the possibilities:

 

As you can see, the columns here represent the actual class—that is, whether an image actually has a dog or a cat. The rows represent the predicted class—that is, whether the model concludes that an image has a dog or a cat. When the predicted class matches the actual class (e.g., the model says the image has a cat and the image does indeed have a cat), this is a correct classification.
 
 


The key is to look at the diagonals. If the upper left and lower right cells are high relative to the others, then the model is making more correct classifications than incorrect classifications:

 
Whereas if the upper right and lower left cells are comparatively higher, the model is making more incorrect classifications:

 

This type of table is called a confusion matrix. A confusion matrix gets its name from the fact that it is easy to see whether the model is getting confused and misclassifying the data.

You will often see the confusion matrix represented in a more general, abstract form that uses the terms positive and negative:

 
•	
•	True positives are the positive cases that are correctly predicted as positive by the model
•	False positives are the negative cases that are incorrectly predicted as positive by the model
•	True negatives are the negative cases that are correctly predicted as negative by the model
•	False negatives are the positive cases that are incorrectly predicted as negative by the model
 

We can construct several different very useful metrics from a confusion matrix—and that's what we'll look at next.

3.25 Evaluation Metrics for Classification

https://photos.app.goo.gl/qqYddgejMv3j7NCR9

 

Model Evaluation Charts

https://photos.app.goo.gl/1d7VhuHYnPibWBBK9


 

3.26 Prelaunch Lab

3.27 Evaluation Metrics for Regression

If you recall, classification yields discrete outputs (e.g., cat vs dog or positive vs. negative), while regression yields continuous, numerical outputs (e.g., 3.229, 23 minutes, $17.78).

Not surprisingly then, we need a different set of metrics for evaluating regression models. Let's have a look.

https://photos.app.goo.gl/4xZkuUP4jg3huM1Y8

Again, note that with regression metrics, we are using functions that in some way calculate the numerical difference between the predicted vs. expected values.

https://photos.app.goo.gl/SHT2rBHhVm3WPzvRA

 


3.28 Lab: Train and Evaluate a Model

Lab Overview
Azure Machine Learning designer (preview) gives you a cloud-based interactive, visual workspace that you can use to easily and quickly prep data, train and deploy machine learning models. It supports Azure Machine Learning compute, GPU or CPU. Machine Learning designer also supports publishing models as web services on Azure Kubernetes Service that can easily be consumed by other applications.
In this lab, we will be using the Flight Delays data set that is enhanced with the weather data. Based on the enriched dataset, we will learn to use the Azure Machine Learning Graphical Interface to process data, build, train, score, and evaluate a classification model to predict if a particular flight will be delayed by 15 minutes or more. To train the model, we will use Azure Machine Learning Compute resource. We will do all of this from the Azure Machine Learning designer without writing a single line of code.
Exercise 1: Register Dataset with Azure Machine Learning studio

Task 1: Upload Dataset
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	From the studio, select Datasets, + Create dataset, From web files. This will open the Create dataset from web files dialog on the right.
 
5.	In the Web URL field provide the following URL for the training data file:
https://introtomlsampledata.blob.core.windows.net/data/flightdelays/flightdelays.csv
6.	Provide flightdelays as the Name, leave the remaining values at their defaults and select Next.

 

Task 2: Preview Dataset
1.	On the Settings and preview panel, set the column headers drop down to All files have same headers.
2.	Review the dataset and then select Next
 
Task 3: Select Columns
1.	Select columns from the dataset to include as part of your training data. Leave the default selections and select Next
 
Task 4: Create Dataset
1.	Confirm the dataset details and select Create
 
Exercise 2: Create New Training Pipeline
Task 1: Open Pipeline Authoring Editor
1.	From the studio, select Designer, +. This will open a visual pipeline authoring editor.
 
Task 2: Setup Compute Target
1.	In the settings panel on the right, select Select compute target.
 
2.	In the Set up compute target editor, select the available compute, and then select Save.
Note: If you are facing difficulties in accessing pop-up windows or buttons in the user interface, please refer to the Help section in the lab environment.
 
Task 3: Add Dataset
1.	Select Datasets section in the left navigation. Next, select My Datasets, flightdelays and drag and drop the selected dataset on to the canvas.
 
Task 4: Split Dataset
1.	We will split the dataset such that months prior to October will be used for model training and months October to December will be used for model testing.
2.	Select Data Transformation section in the Modules. Follow the steps outlined below:
1.	Select the Split Data prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Splitting mode: Relative Expression
4.	Relational expression: \”Month” < 10 (please make sure that the expression is provided exactly as seen in the image below, starting with backslash.)
5.	Connect the Dataset to the Split Data module
 

Note that you can submit the pipeline at any point to peek at the outputs and activities. Running pipeline also generates metadata that is available for downstream activities such selecting column names from a list in selection dialogs. Please refer ahead to Exercise 3, Task 1, Step 2 on details of submitting the pipeline. It can take up to 5-10 minutes to run the pipeline.
Task 5: Select Columns in Dataset
1.	Select Data Transformation section in the left navigation. Follow the steps outlined below:
1.	Select the Select Columns in Dataset prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the first output of the Split Data module to the Select Columns in Dataset module
4.	Select Edit column link to open the Select columns` editor
 
2.	In the Select columns editor, follow the steps outlined below:
1.	Include: All columns
2.	Select +
3.	Exclude: Column names, provide the following column names to exclude: Month, Year, Year_R, Timezone, Timezone_R
4.	Select Save
 
Task 6: Initialize Classification Model
1.	Select Machine Learning Algorithms section in the left navigation. Follow the steps outlined below:
1.	Select the Two-Class Logistic Regression prebuilt module
2.	Drag and drop the selected module on to the canvas
 
Task 7: Setup Train Model Module
1.	Select Model Training section in the left navigation. Follow the steps outlined below:
1.	Select the Train Model prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the Two-Class Logistic Regression module to the first input of the Train Model module
4.	Connect the Select Columns in Dataset module to the second input of the Train Model module
5.	Select the Edit column link to open the Label column editor
 
2.	The Label column editor allows you to specify your Label or Target column. Type in the label column name ArrDel15 and then select Save.
 
Task 8: Setup Score Model Module
1.	Select Model Scoring & Evaluation section in the left navigation. Follow the steps outlined below:
1.	Select the Score Model prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the Train Model module to the first input of the Score Model module
4.	Connect the second output of the Split Data module to the second input of the Score Model module
 
Note that Split Data module will feed data for both model training and model scoring.
Task 9: Setup Evaluate Model Module
1.	Select Model Scoring & Evaluation section in the left navigation. Follow the steps outlined below:
1.	Select the Evaluate Model prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the Score Model module to the first input of the Evaluate Model module
 
Exercise 3: Submit Training Pipeline
Task 1: Create Experiment and Submit Pipeline
1.	Select Submit to open the Setup pipeline run editor.
 
Please note that the button name in the UI is changed from Run to Submit.
2.	In the Setup pipeline run editor, select Experiment, Create new and provide New experiment name: flight-delay, and then select Submit.
 
3.	Wait for pipeline run to complete. It will take around 10 minutes to complete the run.
4.	While you wait for the model training to complete, you can learn more about the evaluation metrics for the classification algorithm used in this lab by selecting Metrics for classification models.
Exercise 4: Visualize the Evaluation Results
Task 1: Open the Result Visualization Dialog
1.	Select Evaluate Model, Outputs, Visualize to open the Evaluate Model result visualization dialog.
 
Task 2: Evaluate Model Performance
1.	Evaluate the model performance by reviewing the various evaluation curves, such as ROC curve, Precision-recall curve, and Lift curve.
 
2.	Scroll down to review the following:
1.	Review the key metrics for classifiers: Accuracy, Precision, Recall, F1 Score, and AUC
2.	Review the binary classifier’s Confusion Matrix
 
Next Steps
Congratulations! You have trained and evaluated your first classification machine learning model. You can continue to experiment in the environment but are free to close the lab environment tab and return to the Udacity portal to continue with the lesson.
3.29 Walkthrough: Train and Evaluate a Model

https://photos.app.goo.gl/P96uPvFm1YUQn8Bd7

3.30 Prelaunch Lab
3.31 Strength in Numbers

https://photos.app.goo.gl/PdfeTFpRguM5BUXp8

Remember, no matter how well-trained an individual model is, there is still a significant chance that it could perform poorly or produce incorrect results. Rather than relying on a single model, you can often get better results by training multiple models or using multiple algorithms and in some way capturing the collective results. As we mentioned, there are two main approaches to this: Ensemble learning and automated machine learning. Let's have a closer look at each of them.

Ensemble Learning

https://photos.app.goo.gl/X9jg6XAsvNbCqMwR6

Remember, ensemble learning combines multiple machine learning models to produce one predictive model. There are three main types of ensemble algorithms:

Bagging or bootstrap aggregation
•	Helps reduce overfitting for models that tend to have high variance (such as decision trees)
•	Uses random subsampling of the training data to produce a bag of trained models.
•	The resulting trained models are homogeneous
•	The final prediction is an average prediction from individual models
Boosting
•	Helps reduce bias for models.
•	In contrast to bagging, boosting uses the same input data to train multiple models using different hyperparameters.
•	Boosting trains model in sequence by training weak learners one by one, with each new learner correcting errors from previous learners
•	The final predictions are a weighted average from the individual models
Stacking
•	Trains a large number of completely different (heterogeneous) models
•	Combines the outputs of the individual models into a meta-model that yields more accurate predictions

Strength in Variety: Automated ML

https://photos.app.goo.gl/ERLQBCZNxteb3iWi6

Automated machine learning, like the name suggests, automates many of the iterative, time-consuming, tasks involved in model development (such as selecting the best features, scaling features optimally, choosing the best algorithms, and tuning hyperparameters). Automated ML allows data scientists, analysts, and developers to build models with greater scale, efficiency, and productivity—all while sustaining model quality. 

Now that we've talked about the concepts underlying ensemble learning and automated machine learning, let's get some hands-on practice with both of these approaches in Azure Machine Leaning Studio.

3.32 Lab: Train a Two-Class Boosted Decision Tree
Lab Overview
Azure Machine Learning designer (preview) gives you a cloud-based interactive, visual workspace that you can use to easily and quickly prep data, train and deploy machine learning models. It supports Azure Machine Learning compute, GPU or CPU. Machine Learning designer also supports publishing models as web services on Azure Kubernetes Service that can easily be consumed by other applications.
In this lab, we will be using the Flight Delays data set that is enhanced with the weather data. Based on the enriched dataset, we will learn to use the Azure Machine Learning Graphical Interface to process data, build, train, score, and evaluate a classification model to predict if a particular flight will be delayed by 15 minutes or more. The classification algorithm used in this lab will be the ensemble algorithm: Two-Class Boosted Decision Tree. To train the model, we will use Azure Machine Learning Compute resource. We will do all of this from the Azure Machine Learning designer without writing a single line of code.
Exercise 1: Register Dataset with Azure Machine Learning studio
Task 1: Upload Dataset
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	From the studio, select Datasets, + Create dataset, From web files. This will open the Create dataset from web files dialog on the right.
 
5.	In the Web URL field provide the following URL for the training data file:
 https://introtomlsampledata.blob.core.windows.net/data/flightdelays/flightdelays.csv
6.	Provide flightdelays as the Name, leave the remaining values at their defaults and select Next.
 
Task 2: Preview Dataset
1.	On the Settings and preview panel, set the column headers drop down to All files have same headers.
2.	Review the dataset and then select Next
 
Task 3: Select Columns
1.	Select columns from the dataset to include as part of your training data. Leave the default selections and select Next
 
Task 4: Create Dataset
1.	Confirm the dataset details and select Create
 
Exercise 2: Create New Training Pipeline
Task 1: Open Pipeline Authoring Editor
1.	From the studio, select Designer, +. This will open a visual pipeline authoring editor.
 
Task 2: Setup Compute Target
1.	In the settings panel on the right, select Select compute target.
 
2.	In the Set up compute target editor, select the available compute, and then select Save.
Note: If you are facing difficulties in accessing pop-up windows or buttons in the user interface, please refer to the Help section in the lab environment.
 
Task 3: Add Dataset
1.	Select Datasets section in the left navigation. Next, select My Datasets, flightdelays and drag and drop the selected dataset on to the canvas.
 
Task 4: Split Dataset
1.	We will split the dataset such that months prior to October will be used for model training and months October to December will be used for model testing.
2.	Select Data Transformation section in the left navigation. Follow the steps outlined below:
1.	Select the Split Data prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Splitting mode: Relative Expression
4.	Relational expression: \”Month” < 10 (please make sure that the expression is provided exactly as seen in the image below, starting with backslash.)
5.	Connect the Dataset to the Split Data module
 
Note that you can submit the pipeline at any point to peek at the outputs and activities. Running pipeline also generates metadata that is available for downstream activities such selecting column names from a list in selection dialogs.
Task 5: Select Columns in Dataset
1.	Select Data Transformation section in the left navigation. Follow the steps outlined below:
1.	Select the Select Columns in Dataset prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the first output of the Split Data module to the Select Columns in Dataset module
4.	Select Edit column link to open the Select columns` editor
 
2.	In the Select columns editor, follow the steps outlined below:
1.	Include: All columns
2.	Select +
3.	Exclude: Column names, provide the following column names to exclude: Month, Year, Year_R, Timezone, Timezone_R
4.	Select Save
 
Task 6: Initialize Classification Model
1.	Select Machine Learning Algorithms section in the left navigation. Follow the steps outlined below:
1.	Select the Two-Class Boosted Decision Tree prebuilt module
2.	Drag and drop the selected module on to the canvas
 
Task 7: Setup Train Model Module
1.	Select Model Training section in the left navigation. Follow the steps outlined below:
1.	Select the Train Model prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the Two-Class Boosted Decision Tree module to the first input of the Train Model module
4.	Connect the Select Columns in Dataset module to the second input of the Train Model module
5.	Select the Edit column link to open the Label column editor
 
2.	The Label column editor allows you to specify your Label or Target column. Type in the label column name ArrDel15 and then select Save.
 
Task 8: Setup Score Model Module
1.	Select Model Scoring & Evaluation section in the left navigation. Follow the steps outlined below:
1.	Select the Score Model prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the Train Model module to the first input of the Score Model module
4.	Connect the second output of the Split Data module to the second input of the Score Model module
 
Note that Split Data module will feed data for both model training and model scoring.
Task 9: Setup Evaluate Model Module
1.	Select Model Scoring & Evaluation section in the left navigation. Follow the steps outlined below:
1.	Select the Evaluate Model prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the Score Model module to the first input of the Evaluate Model module
 
Exercise 3: Submit Training Pipeline
Task 1: Create Experiment and Submit Pipeline
1.	Select Submit to open the Setup pipeline run editor.
 
Please note that the button name in the UI is changed from Run to Submit.
2.	In the Setup pipeline run editor, select Experiment, Create new and provide New experiment name: flight-delay, and then select Submit.
 
3.	Wait for pipeline run to complete. It will take around 10 minutes to complete the run.
4.	While you wait for the model training to complete, you can learn more about the classification algorithm used in this lab by selecting Two-Class Boosted Decision Tree.
Exercise 4: Visualize the Evaluation Results
Task 1: Open the Result Visualization Dialog
1.	Select Evaluate Model, Outputs, Visualize to open the Evaluate Model result visualization dialog.
 
Task 2: Evaluate Model Performance
1.	Evaluate the model performance by reviewing the various evaluation curves, such as ROC curve, Precision-recall curve, and Lift curve.
 
2.	Scroll down to review the following:
1.	Review the key metrics for classifiers: Accuracy, Precision, Recall, F1 Score, and AUC
2.	Review the binary classifier’s Confusion Matrix
 
Next Steps
Congratulations! You have trained and evaluated your first ensemble machine learning model. You can continue to experiment in the environment but are free to close the lab environment tab and return to the Udacity portal to continue with the lesson.


3.33 Walkthrough: Train a Two-Class Boosted Decision Tree

https://photos.app.goo.gl/yAXiP3FPMvwCbMSe9

3.34 Prelaunch Lab


3.35 Lab: Train a Simple Classifier with Automated ML

Lab Overview
Automated machine learning picks an algorithm and hyperparameters for you and generates a model ready for deployment. There are several options that you can use to configure automated machine learning experiments.
Configuration options available in automated machine learning:
•	Select your experiment type: Classification, Regression or Time Series Forecasting
•	Data source, formats, and fetch data
•	Choose your compute target
•	Automated machine learning experiment settings
•	Run an automated machine learning experiment
•	Explore model metrics
•	Register and deploy model
You can create and run automated machine learning experiments in code using the Azure ML Python SDK or if you prefer a no code experience, you can also create your automated machine learning experiments in Azure Machine Learning Studio.
In this lab, you learn how to create, run, and explore automated machine learning experiments in the Azure Machine Learning Studio without a single line of code. As part of this lab, we will be using the Flight Delays data set that is enhanced with the weather data. Based on the enriched dataset, we will use automated machine learning to find the best performing classification model to predict if a particular flight will be delayed by 15 minutes or more.

Exercise 1: Register Dataset with Azure Machine Learning studio
Task 1: Upload Dataset
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	From the studio, select Datasets, + Create dataset, From web files. This will open the Create dataset from web files dialog on the right.
 
5.	In the Web URL field provide the following URL for the training data file:
 https://introtomlsampledata.blob.core.windows.net/data/flightdelays/flightdelays.csv
6.	Provide flightdelays-automl as the Name, leave the remaining values at their defaults and select Next.
 
Task 2: Preview Dataset
1.	On the Settings and preview panel, set the column headers drop down to All files have same headers.
2.	Review the dataset and then select Next
 
Task 3: Select Columns
1.	Select columns from the dataset to include as part of your training data. Exclude the following columns: Path, Month, Year, Timezone, Year_R, Timezone_R, and then select Next
 
Task 4: Create Dataset
1.	Confirm the dataset details and select Create
 
Exercise 2: Setup New Automated Machine Learning Experiment
Task 1: Create New Automated Machine Learning Experiment
1.	From the studio home, select Create new, Automated ML run
 
2.	This will open a Create a new automated machine learning experiment page
Task 2: Select Training Data
1.	Select the dataset flightdelays-automl and then select Next
 
Task 3: Create a new Automated ML run
1.	Provide an experiment name: flight-delay
2.	Select target column: ArrDel15
3.	Select compute target: select the available compute
4.	Select Next
 
Task 4: Setup Task type and Settings
1.	Select task type: Classification, and then select View additional configuration settings
 
2.	This will open the Additional configurations dialog.
3.	Provide the following information and then select Save
1.	Primary metric: AUC weighted
2.	Exit criteria, Training job time (hours): 1
3.	Exit criteria, Metric score threshold: 0.7
 
Note that we are setting a metric score threshold to limit the training time. In practice, for initial experiments, you will typically only set the training job time to allow AutoML to discover the best algorithm to use for your specific data.
Exercise 3: Start and Monitor Experiment
Task 1: Start Experiment
1.	Select Finish to start running the experiment
 
Task 2: Monitor Experiment
1.	The experiment will run for about 30 min. Note that most of the time will be spent in the data preparation step and once the data preparation is done, the experiment will take an additional 1-2 minutes to complete.
2.	In the Details tab, observe the run status of the job.
 
3.	Select the Models tab, and observe the various algorithms the AutoML is evaluating. You can also observe the corresponding AUC weighted scores for each algorithm.
 
Note that we have set a metric score threshold to limit the training time. As a result you might see only one algorithm in your models list.
4.	Select Details and wait till the run status becomes Completed.
 
5.	While you wait for the model training to complete, you can learn to view and understand the charts and metrics for your automated machine learning run by selecting Understand automated machine learning results.
Exercise 4: Review Best Model’s Performance
Task 1: Review Best Model Performance
1.	The Details tab shows the Best model summary. Next, select Algorithm name to review the model details.
 
2.	From the Model details tab, to view the various metrics to evaluate the best model performance, select View all other metrics.
 
3.	Review the model performance metrics and then select Close.
 
4.	Next, select Metrics to review the various model performance curves, such as Precision-Recall, ROC, Calibration curve, Gain & Lift curves, and Confusion matrix.
 
Next Steps
Congratulations! You have trained and evaluated your first automated machine learning model. You can continue to experiment in the environment but are free to close the lab environment tab and return to the Udacity portal to continue with the lesson.


3.36 Walkthrough: Train a Simple Classifier with Automated ML

https://photos.app.goo.gl/KcMksK62PozUXWDPA

3.37 Lesson Summary

https://photos.app.goo.gl/tU8xS9XhiuAjQChx6





In this lesson, you've learned to perform the essential data preparation and management tasks involved in machine learning:

•	Data importing and transformation
•	The use of datastores and datasets
•	Versioning
•	Feature engineering
•	Monitoring for data drift

The second major area we covered in this lesson was model training, including:

•	The core model training process
•	Two of the fundamental machine learning models: Classifier and regressor
•	The model evaluation process and relevant metrics

The final part of the lesson focused on how to get better results by using multiple trained models instead of a single one. In this context, you learned about ensemble learning and automated machine learning. You've learned how the two differ, yet apply the same general principle of "strength in numbers". In the process, you trained an ensemble model (a decision forest) and a straightforward classifier using automated Machine Learning.

4 Supervised and Unsupervised Learning

4.1 Lesson Overview

https://photos.app.goo.gl/pxaPKn9T4AR2J2wq8

This lesson covers two of Machine Learning's fundamental approaches: supervised and unsupervised learning.

First, we'll cover supervised learning. Specifically, we'll learn:

•	More about classification and regression, two of the most representative supervised learning tasks
•	Some of the major algorithms involved in supervised learning, as well as how to evaluate and compare their performance
•	How to use automated machine learning to automate the training and selection of classifiers and regressors, and how to use the Designer in Azure Machine Learning Studio to create automated Machine Learning experiments


Next, the lesson will focus on unsupervised learning, including:

•	Its most representative learning task, clustering
•	How unsupervised learning can address challenges like lack of labeled data, the curse of dimensionality, overfitting, feature engineering, and outliers
•	An introduction to representation learning
•	How to train your first clustering model in Azure Machine Learning Studio

4.2 Prelaunch Lab

4.3 Supervised Learning: Classification

The first type of supervised learning that we'll look at is classification. Recall that the main distinguishing characteristic of classification is the type of output it produces:

In a classification problem, the outputs are categorical or discrete.

Within this broad definition, there are several main approaches, which differ based on how many classes or categories are used, and whether each output can belong to only one class or multiple classes. Let's have a look.

https://photos.app.goo.gl/x1htkBdpdcfgRaus7

Some of the most common types of classification problems include:

•	Classification on tabular data: The data is available in the form of rows and columns, potentially originating from a wide variety of data sources.
•	Classification on image or sound data: The training data consists of images or sounds whose categories are already known.
•	Classification on text data: The training data consists of texts whose categories are already known.

As we discussed in a previous lesson, machine learning requires numerical data. This means that with images, sound, and text, several steps need to be performed during the preparation phase to transform the data into numerical vectors that can be accepted by the classification algorithms.

Categories of Algorithms
https://photos.app.goo.gl/TEUBr8Fogty8SL2s6

 

 

4.4 Lab: Two-Class Classifiers Performance

Lab Overview
Azure Machine Learning designer (preview) gives you a cloud-based interactive, visual workspace that you can use to easily and quickly prep data, train and deploy machine learning models. It supports Azure Machine Learning compute, GPU or CPU. Machine Learning designer also supports publishing models as web services on Azure Kubernetes Service that can easily be consumed by other applications.
In this lab, we will be compare the performance of two binary classifiers: Two-Class Boosted Decision Tree and Two-Class Logistic Regression for predicting customer churn. The goal is to run an expensive marketing campaign for high risk customers; thus, the precision metric is going to be key in evaluating performance of these two algorithms. We will do all of this from the Azure Machine Learning designer without writing a single line of code.
Exercise 1: Create Training Pipeline
Task 1: Open Sample 5: Binary Classification – Customer Relationship Prediction
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	From the studio, select Designer, Sample 5: Binary Classification – Customer Relationship Prediction.
 
Task 2: Setup Compute Target
1.	In the settings panel on the right, select Select compute target.
 
2.	In the Set up compute target editor, select the available compute, and then select Save.
Note: If you are facing difficulties in accessing pop-up windows or buttons in the user interface, please refer to the Help section in the lab environment.
 
Task 3: Delete Pipeline Modules
1.	From the right-hand-side of the pipeline, select the Two-Class Boosted Decision Tree module and then select the Delete Icon.
 
2.	From the right-hand-side of the pipeline, select the SMOTE module and then select the Delete Icon.
 
Task 4: Setup the Two-Class Logistic Regression Module
1.	Select Machine Learning Algorithms section in the left navigation. Follow the steps outlined below:
1.	Select the Two-Class Logistic Regression prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the Two-Class Logistic Regression module to the first input of the Train Model module
4.	Connect the first output of the Split Data module to the second input of the Train Model module
 
Exercise 2: Submit Training Pipeline
Task 1: Create Experiment and Submit Pipeline
1.	Select Submit to open the Setup pipeline run editor.
 
Please note that the button name in the UI is changed from Run to Submit.
2.	In the Setup pipeline run editor, select Experiment, Create new and provide New experiment name: Churn-Predictor, and then select Submit.
 
3.	Wait for pipeline run to complete. It will take around 5 minutes to complete the run.
4.	While you wait for the model training to complete, you can learn more about the evaluation metrics for the classification algorithm used in this lab by selecting Metrics for classification models.
Exercise 3: Compare Model Performance
Task 1: Open Evaluation Results for Two-Class Boosted Decision Tree
1.	From the left-hand-side of the pipeline, select Evaluate Model, Outputs, Visualize to open the Evaluate Model result visualization dialog for the Two-Class Boosted Decision Tree module.
 
Task 2: Evaluate Two-Class Boosted Decision Tree Performance
1.	Scroll down to review model performance metrics for Two-Class Boosted Decision Tree. Observe that the Precision value is around 0.7.
 
Task 3: Open Evaluation Results for Two-Class Logistic Regression
1.	From the right-hand-side of the pipeline, select Evaluate Model, Outputs, Visualize to open the Evaluate Model result visualization dialog for the Two-Class Logistic Regression module.
 
Task 4: Evaluate Two-Class Logistic Regression Performance
1.	Scroll down to review model performance metrics for Two-Class Logistic Regression. Observe that the Precision value is around 0.3.
 
Task 5: Conclusion
1.	Based on the primary performance metric, Precision, it shows that the Two-Class Boosted Decision Tree algorithm outperforms the Two-Class Logistic Regression algorithm.
Next Steps
Congratulations! You have trained and compared performance of two different classification machine learning models. You can continue to experiment in the environment but are free to close the lab environment tab and return to the Udacity portal to continue with the lesson.
4.5 Walkthrough: Two-Class Classifiers Performance

https://photos.app.goo.gl/uUz94vAEiXbdpurP8

4.6 Prelaunch Lab



4.7 Multi-Class Algorithms

https://photos.app.goo.gl/bNXUPd8KgXevxobFA

 












4.8 Lab: Multi-Class Classifiers Performance

Lab Overview
Azure Machine Learning designer (preview) gives you a cloud-based interactive, visual workspace that you can use to easily and quickly prep data, train and deploy machine learning models. It supports Azure Machine Learning compute, GPU or CPU. Machine Learning designer also supports publishing models as web services on Azure Kubernetes Service that can easily be consumed by other applications.
In this lab, we will be compare the performance of two different multiclass classification approaches: Two-Class Support Vector Machine used with One-vs-All Multiclass module vs Multiclass Decision Forest. We will apply the two approaches for the letter recognition problem and compare their performance. We will do all of this from the Azure Machine Learning designer without writing a single line of code.
Exercise 1: Create Training Pipeline
Task 1: Open Sample 12: Multiclass Classification - Letter Recognition
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	From the studio, select Designer, Show more samples.
 
5.	Select Sample 12: Multiclass Classification - Letter Recognition.
 
Task 2: Setup Compute Target
1.	In the settings panel on the right, select Select compute target.
 
2.	In the Set up compute target editor, select the available compute, and then select Save.
Note: If you are facing difficulties in accessing pop-up windows or buttons in the user interface, please refer to the Help section in the lab environment.
 
Exercise 2: Submit Training Pipeline
Task 1: Create Experiment and Submit Pipeline
1.	Select Submit to open the Setup pipeline run editor.
 
Please note that the button name in the UI is changed from Run to Submit.
2.	In the Setup pipeline run editor, select Experiment, Create new and provide New experiment name: letter-recognition, and then select Submit.
 
3.	Wait for pipeline run to complete. It will take around 10 minutes to complete the run.
4.	While you wait for the model training to complete, you can learn more about the One-vs-All Multiclass module used in this lab by selecting One-vs-All Multiclass.
Exercise 3: Compare Model Performance
Task 1: Open Evaluation Results
1.	Select Evaluate Model, Outputs, Visualize to open the Evaluate Model result visualization dialog.
 
Task 2: Compare Performance Metrics
1.	Select the regression performance metric Overall_Accuracy and compare performance of the two algorithms: Two-Class Support Vector Machine and Multiclass Decision Forest.
 
Task 3: Conclusion
1.	The Two-Class Support Vector Machine algorithm is extended for multiclass classification problem by using the One-vs-All Multiclass module.
2.	As you can observe that the native multiclass algorithm Multiclass Decision Forest outperforms the Two-Class Support Vector Machine across all key performance metrics.
3.	One recommendation for next steps is to increase the Number of iterations parameter for the Two-Class Support Vector Machine module to an higher value like 100 and observe its impact on the performance metrics.
Next Steps
Congratulations! You have trained and compared performance of two different multiclass classification machine learning models. You can continue to experiment in the environment but are free to close the lab environment tab and return to the Udacity portal to continue with the lesson.

4.9 Walkthrough: Multi-Class Classifiers Performance

https://photos.app.goo.gl/PgjofAH4gZwfT7T2A

4.10 Prelaunch Lab

4.11 Lab: Train a Classifier Using Automated Machine Learning
Lab Overview
Automated machine learning picks an algorithm and hyperparameters for you and generates a model ready for deployment. There are several options that you can use to configure automated machine learning experiments.
Configuration options available in automated machine learning:
•	Select your experiment type: Classification, Regression or Time Series Forecasting
•	Data source, formats, and fetch data
•	Choose your compute target
•	Automated machine learning experiment settings
•	Run an automated machine learning experiment
•	Explore model metrics
•	Register and deploy model
You can create and run automated machine learning experiments in code using the Azure ML Python SDK or if you prefer a no code experience, you can also create your automated machine learning experiments in Azure Machine Learning Studio.
In this lab, we will use Automated Machine Learning to find the best performing binary classification model for predicting customer churn. We will do all of this from the Azure Machine Learning Studio without writing a single line of code.
Exercise 1: Register Dataset with Azure Machine Learning studio
Task 1: Upload Dataset
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	From the studio, select Datasets, + Create dataset, From web files. This will open the Create dataset from web files dialog on the right.
 
5.	In the Web URL field provide the following URL for the training data file:
 https://introtomlsampledata.blob.core.windows.net/data/crm-churn/crm-churn.csv
6.	Provide CRM-Churn as the Name, leave the remaining values at their defaults and select Next.
 
Task 2: Preview Dataset
1.	On the Settings and preview panel, set the column headers drop down to All files have same headers.
2.	Review the dataset and then select Next
 
Task 3: Select Columns
1.	Keep the default selections, and select Next
 
Task 4: Create Dataset
1.	Confirm the dataset details and select Create
 
Exercise 2: Setup New Automated Machine Learning Experiment
Task 1: Create New Automated Machine Learning Experiment
1.	From the studio home, select Create new, Automated ML run
 
2.	This will open a Create a new automated machine learning experiment page
Task 2: Select Training Data
1.	Select the dataset CRM-Churn and then select Next
 
Task 3: Create a new Automated ML run
1.	Provide an experiment name: Churn-Predictor
2.	Select target column: Col1
3.	Select compute target: select the available compute
4.	Select Next
 
Task 4: Setup Task type and Settings
1.	Select task type: Classification, and then select View additional configuration settings
 
2.	This will open the Additional configurations dialog.
3.	Provide the following information and then select Save
1.	Primary metric: AUC weighted
2.	Exit criteria, Metric score threshold: 0.707
 
Note that we are setting a metric score threshold to limit the training time. In practice, for initial experiments, you will typically only set the training job time to allow AutoML to discover the best algorithm to use for your specific data.
Exercise 3: Start and Monitor Experiment
Task 1: Start Experiment
1.	Select Finish to start running the experiment
 
Task 2: Monitor Experiment
1.	The experiment will run for about 5 min
2.	In the Details tab, observe the run status of the job.
 
3.	Wait till the run status becomes Completed.
 
4.	While you wait for the model training to complete, you can learn to view and understand the charts and metrics for your automated machine learning run by selecting Understand automated machine learning classification results.
Exercise 4: Review Best Model’s Performance
Task 1: Review Best Model Performance
1.	From the Details tab review the best model’s Algorithm name and its corresponding AUC weighted score. Next, select the best model’s Algorithm name
 
2.	Select View all other metrics to review the various Run Metrics to evaluate the model performance. Next, select Metrics
 
3.	Select accuracy_table, Chart to review the various model performance curves, such as Precision-Recall, ROC, Calibration curve, and Gain & Lift curves.
 
Next Steps
Congratulations! You have trained and evaluated a binary classification model using automated machine learning. You can continue to experiment in the environment but are free to close the lab environment tab and return to the Udacity portal to continue with the lesson.


4.12 Walkthrough: Train a Classifier Using Automated ML

https://photos.app.goo.gl/RzhQHDwBaqVKg8H29

4.13 Prelaunch Lab

4.14 Supervised Learning: Regression

The first type of supervised learning that we'll look at is classification. Again, the main distinguishing characteristic of regression is the type of output it produces:

In a regression problem, the output is numerical or continuous.

Introduction to Regression

https://photos.app.goo.gl/dKQiUME9AG4pp3Vv5

Common types of regression problems include:

•	Regression on tabular data: The data is available in the form of rows and columns, potentially originating from a wide variety of data sources.
•	Regression on image or sound data: Training data consists of images/sounds whose numerical scores are already known. Several steps need to be performed during the preparation phase to transform images/sounds into numerical vectors accepted by the algorithms.
•	Regression on text data: Training data consists of texts whose numerical scores are already known. Several steps need to be performed during the preparation phase to transform text into numerical vectors accepted by the algorithms.

Categories of Algorithms

https://photos.app.goo.gl/NfWfaGGwZGiYnZVJ7

Common machine learning algorithms for regression problems include:

•	Linear Regression
•	Fast training, linear model

•	Decision Forest Regression
•	Accurate, fast training times

•	Neural Net Regression
•	Accurate, long training times

 
 

4.15 Lab: Regressors Performance

Lab Overview
Azure Machine Learning designer (preview) gives you a cloud-based interactive, visual workspace that you can use to easily and quickly prep data, train and deploy machine learning models. It supports Azure Machine Learning compute, GPU or CPU. Machine Learning designer also supports publishing models as web services on Azure Kubernetes Service that can easily be consumed by other applications.
In this lab, we will be compare the performance of two regression algorithms: Boosted Decision Tree Regression and Neural Net Regression for predicting automobile prices. We will do all of this from the Azure Machine Learning designer without writing a single line of code.
Exercise 1: Create Training Pipeline
Task 1: Open Sample 2: Regression - Automobile Price Prediction (Compare algorithms)
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	From the studio, select Designer, Sample 2: Regression - Automobile Price Prediction (Compare algorithms).
 
Task 2: Setup Compute Target
1.	In the settings panel on the right, select Select compute target.
 
2.	In the Set up compute target editor, select the available compute, and then select Save.
Note: If you are facing difficulties in accessing pop-up windows or buttons in the user interface, please refer to the Help section in the lab environment.
 
Task 3: Delete Pipeline Modules
1.	From the right-hand-side of the pipeline, select the Decision Forest Regression module and then select the Delete Icon.
 
Task 4: Setup the Neural Net Regression Module
1.	Select Machine Learning Algorithms section in the left navigation. Follow the steps outlined below:
1.	Select the Neural Net Regression prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Set Number of hidden nodes to 1000
4.	Set Learning rate to 0.0001
5.	Set Number of learning iterations to 10000
6.	Set Random number seed to 139
7.	Connect the Neural Net Regression module to the first input of the Train Model module
 
Exercise 2: Submit Training Pipeline
Task 1: Create Experiment and Submit Pipeline
1.	Select Submit to open the Setup pipeline run editor.
 
Please note that the button name in the UI is changed from Run to Submit.
2.	In the Setup pipeline run editor, select Experiment, Create new and provide New experiment name: automobile-price-prediction, and then select Submit.
 
3.	Wait for pipeline run to complete. It will take around 10 minutes to complete the run.
4.	While you wait for the model training to complete, you can learn more about the evaluation metrics for the regression algorithm used in this lab by selecting Metrics for regression models.
Exercise 3: Compare Model Performance
Task 1: Open Evaluation Results
1.	Select Evaluate Model, Outputs, Visualize to open the Evaluate Model result visualization dialog.
 
Task 2: Compare Performance Metrics
1.	Select the regression performance metric Root_Mean_Squared_Error and compare performance of the two algorithms: Boosted Decision Tree Regression and Neural Net Regression. Note that smaller value for Root_Mean_Squared_Error implies better performance.
 
Task 3: Conclusion
1.	Based on the performance metric, Root_Mean_Squared_Error, it shows that the Boosted Decision Tree Regression algorithm outperforms the Neural Net Regression algorithm. One recommendation for next steps is to tune the hyperparameters for the Neural Net Regression module to see if we can improve its performance.
Next Steps
Congratulations! You have trained and compared performance of two different regression machine learning models. You can continue to experiment in the environment but are free to close the lab environment tab and return to the Udacity portal to continue with the lesson.
4.16 Walkthrough: Regressors Performance
https://photos.app.goo.gl/CB7wdtXhDGTQQQDA6

4.17 Prelaunch Lab
4.18 Automate the Training of Regressors
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 
4.19 Lab: Train a Regressor using Automated Machine Learning

Lab Overview
Automated machine learning picks an algorithm and hyperparameters for you and generates a model ready for deployment. There are several options that you can use to configure automated machine learning experiments.
Configuration options available in automated machine learning:
•	Select your experiment type: Classification, Regression or Time Series Forecasting
•	Data source, formats, and fetch data
•	Choose your compute target
•	Automated machine learning experiment settings
•	Run an automated machine learning experiment
•	Explore model metrics
•	Register and deploy model
You can create and run automated machine learning experiments in code using the Azure ML Python SDK or if you prefer a no code experience, you can also create your automated machine learning experiments in Azure Machine Learning Studio.
In this lab, we will use Automated Machine Learning to find the best performing regression model for predicting automobile prices. We will do all of this from the Azure Machine Learning Studio without writing a single line of code.
Exercise 1: Register Dataset with Azure Machine Learning studio
Task 1: Upload Dataset
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	From the studio, select Datasets, + Create dataset, From web files. This will open the Create dataset from web files dialog on the right.
 
5.	In the Web URL field provide the following URL for the training data file:
 https://introtomlsampledata.blob.core.windows.net/data/automobile-price/automobile-price.csv
6.	Provide Automobile-Price as the Name, leave the remaining values at their defaults and select Next.
 
Task 2: Preview Dataset
1.	On the Settings and preview panel, set the column headers drop down to All files have same headers.
2.	Review the dataset and then select Next
 
Task 3: Select Columns
1.	Keep the default selections, and select Next
 
Task 4: Create Dataset
1.	Confirm the dataset details and select Create
 
Exercise 2: Setup New Automated Machine Learning Experiment
Task 1: Create New Automated Machine Learning Experiment
1.	From the studio home, select Create new, Automated ML run
 
2.	This will open a Create a new automated machine learning experiment page
Task 2: Select Training Data
1.	Select the dataset Automobile-Price and then select Next
 
Task 3: Create a new Automated ML run
1.	Provide an experiment name: automobile-price-prediction
2.	Select target column: price
3.	Select compute target: select the available compute
4.	Select Next
 
Task 4: Setup Task type and Settings
1.	Select task type: Regression, and then select View additional configuration settings
 
2.	This will open the Additional configurations dialog.
3.	Provide the following information and then select Save
1.	Primary metric: Normalized root mean squared error
2.	Exit criteria, Metric score threshold: 0.056
 
Note that we are setting a metric score threshold to limit the training time. In practice, for initial experiments, you will typically only set the training job time to allow AutoML to discover the best algorithm to use for your specific data.
Exercise 3: Start and Monitor Experiment
Task 1: Start Experiment
1.	Select Finish to start running the experiment
 
Task 2: Monitor Experiment
1.	The experiment will run for about 10 min.
2.	In the Details tab, observe the run status of the job.
 
3.	Select the Models tab, and observe the various algorithms the AutoML is evaluating. You can also observe the corresponding Normalized root mean squared error scores for each algorithm.
 
4.	Select Details and wait till the run status becomes Completed.
 
5.	While you wait for the model training to complete, you can learn more about how Automated Machine Learning offers preprocessing and data guardrails automatically by selecting Automatic featurization.
Exercise 4: Review Best Model’s Performance
Task 1: Review Best Model Performance
1.	From the Details tab review the best model’s Algorithm name and its corresponding Normalized root mean squared error score. Next, select the best model’s Algorithm name
 
2.	Select View all other metrics to review the various Run Metrics to evaluate the model performance. Next, select Metrics
 
3.	Select predicted_true, Chart to review the Predicted vs. True curve.
 
Next Steps
Congratulations! You have trained and evaluated a regression model using automated machine learning. You can continue to experiment in the environment but are free to close the lab environment tab and return to the Udacity portal to continue with the lesson.

4.20 Walkthrough: Train a Regressor using Automated Machine Learning

https://photos.app.goo.gl/w7K61HeokNgMjenb8

4.21 Unsupervised Learning

https://photos.app.goo.gl/th7bfWQFABVgpuad6

All of the algorithms we have looked at so far are examples of supervised learning, in which the training data is labeled. For example, if we are training a classifier to recognize an image of a cat, we might have some images in our training dataset that we already know have cats—and are labeled as such.

But the cost of obtaining labeled data can be high. So now let's turn our attention to the second main type of machine learning explored in this lesson: unsupervised learning.

In unsupervised learning, algorithms learn from unlabeled data by looking for hidden structures in the data.

https://photos.app.goo.gl/SXMABuEdHzGY6oBz7

Obtaining unlabeled data is comparatively inexpensive and unsupervised learning can be used to uncover very useful information in such data. For example, we can use clustering algorithms to discover implicit grouping within the data, or use association algorithms to discover hidden rules that are governing the data (e.g., people who buy product A also tend to buy product B).

Types of Unsupervised Machine Learning

https://photos.app.goo.gl/YpZA56sb8MxpTpy77


 

4.22 Semi-Supervised Learnining

Sometimes fully labeled data cannot be obtained, or is too expensive—but at the same time, it may be possible to get partially labeled data. This is where semi-supervised learning is useful.

Semi-supervised learning combines the supervised and unsupervised approaches; typically it involves having small amounts of labeled data and large amounts of unlabeled data.

https://photos.app.goo.gl/pEQyttMDmbq6YCn99

 

4.23 Prelaunch Lab

4.24 Clustering

On this page, we'll discuss the unsupervised approach of clustering in more detail.

As the name suggests, clustering is the problem of organizing entities from the input data into a finite number of subsets or clusters; the goal is to maximize both intra-cluster similarity and inter-cluster differences.

Clustering Algorithms

https://photos.app.goo.gl/NDNrQwbuNAS3rd9o9

K-Means Clustering
 

https://photos.app.goo.gl/uUWtiJ6n36K8MgmK8

 

4.25 Lab: Train a Simple Clustering Model
Lab Overview
Azure Machine Learning designer (preview) gives you a cloud-based interactive, visual workspace that you can use to easily and quickly prep data, train and deploy machine learning models. It supports Azure Machine Learning compute, GPU or CPU. Machine Learning designer also supports publishing models as web services on Azure Kubernetes Service that can easily be consumed by other applications.
In this lab, we will be using the Weather Dataset that has weather data for 66 different airports in the USA from April to October 2013. We will cluster the dataset into 5 distinct clusters based on key weather metrics, such as visibility, temperature, dew point, wind speed etc. The goal is to group airports with similar weather conditions. We will do all of this from the Azure Machine Learning designer without writing a single line of code.
Exercise 1: Create New Training Pipeline
Task 1: Open Pipeline Authoring Editor
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	From the studio, select Designer, +. This will open a visual pipeline authoring editor.
 
Task 2: Setup Compute Target
1.	In the settings panel on the right, select Select compute target.
 
2.	In the Set up compute target editor, select the available compute, and then select Save.
Note: If you are facing difficulties in accessing pop-up windows or buttons in the user interface, please refer to the Help section in the lab environment.
 
Task 3: Add Dataset
1.	Select Datasets section in the left navigation. Next, select Samples, Weather Dataset and drag and drop the selected dataset on to the canvas.
 
Task 4: Select Columns in Dataset
1.	Select Modules, Data Transformation section in the left navigation. Follow the steps outlined below:
1.	Select the Select Columns in Dataset prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the Weather Dataset module to the Select Columns in Dataset module
4.	Select Edit column link to open the Select columns editor
 
2.	Note that you can submit the pipeline at any point to peek at the outputs and activities. Running pipeline also generates metadata that is available for downstream activities such selecting column names from a list in selection dialogs.
3.	In the Select columns editor, follow the steps outlined below:
1.	Include: Column indices
2.	Provide column indices: 8, 10-17, 20, 26
3.	Select Save
 
Task 5: Split Data
1.	Select Data Transformation section in the left navigation. Follow the steps outlined below:
1.	Select the Split Data prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the Select Columns in Dataset module to the Split Data module
4.	Fraction of rows in the first output dataset: 0.1
 
Task 6: Normalize Data
1.	Select Data Transformation section in the left navigation. Follow the steps outlined below:
1.	Select the Normalize Data prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the left port ofSplit Data module to the Normalize Data module
4.	Select Edit column link to open the Columns to transform editor
 
2.	In the Columns to transform editor, follow the steps outlined below:
1.	Include: All columns
2.	Select Save
 
Task 7: Initialize K-Means Clustering Model
1.	Select Machine Learning Algorithms section in the left navigation. Follow the steps outlined below:
1.	Select the K-Means Clustering prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Number of centroids: 5
 
Task 8: Setup Train Clustering Model Module
1.	Select Model Training section in the left navigation. Follow the steps outlined below:
1.	Select the Train Clustering Model prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the K-Means Clustering module to the first input of the Train Clustering Model module
4.	Connect the first output of the Normalize Data module to the second input of the Train Clustering Model module
5.	Select the Edit column link to open the Column set editor
 
2.	In the Columns set editor, follow the steps outlined below:
1.	Include: All columns
2.	Select Save
 
Task 9: Setup Assign Data to Clusters Module
1.	Select Model Scoring & Evaluation section in the left navigation. Follow the steps outlined below:
1.	Select the Assign Data to Clusters prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the first output of the Train Clustering Model module to the first input of the Assign Data to Clusters module
4.	Connect the first output of the Normalize Data module to the second input of the Assign Data to Clusters module
 
Exercise 2: Submit Training Pipeline
Task 1: Create Experiment and Submit Pipeline
1.	Select Submit to open the Setup pipeline run editor.
 
2.	In the Setup pipeline run editor, select Experiment, Create new and provide New experiment name: cluster-weather, and then select Submit.
 
3.	Wait for pipeline run to complete. It will take around 10 minutes to complete the run.
4.	While you wait for the model training to complete, you can learn more about the K-Means Clustering algorithm used in this lab by selecting K-Means Clustering.
Exercise 3: Visualize the Clustering Results
Task 1: Open the Visualization Dialog
1.	Select Assign Data to Clusters, Outputs + logs, Visualize to open the Assign Data to Clusters result visualization dialog.
 
Task 2: Evaluate Clustering Results
1.	Scroll to the right and select Assignments column.
2.	In the right-hand-side pane, scroll down to the Visualizations section.
 
3.	From the results you can observe that each row (input) in the dataset is assigned to one of the 5 clusters: 0, 1, 2, 3, or 4. You can also see for each input, how far that input was from the various centroids. The cluster assignment is made based on the shortest distance between the input and cluster centroids. From the bar graph you can see the frequency distribution of all the inputs across the 5 clusters.
Next Steps
Congratulations! You have trained and evaluated your first clustering algorithm. You can continue to experiment in the environment but are free to close the lab environment tab and return to the Udacity portal to continue with the lesson.

4.26 Walkthrough: Train a Simple Clustering Algorithm

https://photos.app.goo.gl/H4rg5vLaXWJtYETY6


 

4.27 Lesson Summary

This lesson covered two of Machine Learning's fundamental approaches: supervised and unsupervised learning.

First, we learned about supervised learning. Specifically, we learned:

•	More about classification and regression, two of the most representative supervised learning tasks
•	Some of the major algorithms involved in supervised learning, as well as how to evaluate and compare their performance
•	How to use the Designer in Azure Machine Learning Studio to build pipelines that train and compare the performance of both binary and multi-class classifiers.
•	How to use automated machine learning to automate the training and selection of classifiers and regressors, and how to use the Designer in Azure Machine Learning Studio to create automated Machine Learning experiments

Next, the lesson focused on unsupervised learning, including:

•	Its most representative learning task, clustering
•	How unsupervised learning can address challenges like lack of labeled data, the curse of dimensionality, overfitting, feature engineering, and outliers
•	An introduction to representation learning
•	How to train your first clustering model in Azure Machine Learning Studio


5 Applications of Machine Learning

5.1 Lesson Overview

https://photos.app.goo.gl/XDMrj4fYM6pU35n56

In this lesson, we will first look at deep learning. You'll learn about:

•	The differences between classical machine learning and deep learning
•	The benefits and applications of Deep Learning
•	How to train your first neural network model

Next, you will learn about some of the most important specialized cases of model training, including:

•	Similarity learning and the basic features of a recommendation engine
•	Text classification and the fundamentals of processing text in machine learning
•	Feature learning, an essential task in feature engineering
•	Anomaly detection
•	Time-series forecasting.

Along the way, you will get practice with several hands-on labs, in which you will train a simple neural network, train a recommendation engine, train a text classifier, and get some experience with forecasting














5.2 Classical Machine Learning vs. Deep Learning

https://photos.app.goo.gl/2gha3V2LWXYmeGkA7

As we just described, Artificial Intelligence (AI) includes Machine Learning (ML), which includes Deep Learning (DL). We can visualize the relationship like this:

 

As the diagram shows, all deep learning algorithms are particular cases of machine learning algorithms—but it's not true that all machine learning algorithms are deep learning algorithms.

A More Detailed Comparison

https://photos.app.goo.gl/PETfj5qKMNpweYkm9


 

5.3 What is Deep Learning?

https://photos.app.goo.gl/GtbKrb7gRd8ymavdA


 

https://photos.app.goo.gl/BCa8Vxf3yWxqjyi7A

The diagram shown in the above video is from Deep Learning, by Ian Goodfellow, Yoshua Bengio, Aaron Courville. The entire book is available for free through their website. In case you'd like to dig more into the comparisons we discussed here, the diagram and accompanying discussion of deep learning can be found in the introduction here.

A Word of Caution About "Neural"

https://photos.app.goo.gl/6sgBMACsXkCAnjGw6

5.4 Characteristics of Deep Learning

https://photos.app.goo.gl/FKgZuzsJaDBjoV3k7



 

5.5 Prelaunch Lab

5.6 Benefits and Applications of Deep Learning

https://photos.app.goo.gl/AcDTyMt3zYUZh3YK9


 
Applications of Deep Learning

https://photos.app.goo.gl/LrNU6XiZ1ij5fXEF7

Now that we've reviewed some of the key characteristics, benefits, and applications of deep learning, let's get some hands-on practice building a simple neural net in Azure Machine Learning Studio.

5.7 Lab: Train a Simple Neural Net

Train a simple neural net model
Although neural networks are widely known for use in deep learning and modeling complex problems such as image recognition, they are easily adapted to regression problems. Any class of statistical models can be termed a neural network if they use adaptive weights and can approximate non-linear functions of their inputs. Thus neural network regression is suited to problems where a more traditional regression model cannot fit a solution.
Neural network regression is a supervised learning method, and therefore requires a tagged dataset, which includes a label column. Because a regression model predicts a numerical value, the label column must be a numerical data type.
Lab Overview
In this lab we will be using a subset of NYC Taxi & Limousine Commission - green taxi trip records available from Azure Open Datasets. The data is enriched with holiday and weather data. Based on the enriched dataset, we will configure the prebuilt Neural Network Regression module to create a regression model using a customizable neural network algorithm. We will train the model by providing the model and the NYC taxi dataset as an input to Train Model. The trained model can then be used to predict NYC taxi fares. We will do all of this from the Azure Machine Learning designer without writing a single line of code.
Exercise 1: Register Dataset with Azure Machine Learning studio
Task 1: Upload Dataset
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	From the studio, select Datasets, + Create dataset, From web files. This will open the Create dataset from web files dialog on the right.
 
5.	In the Web URL field provide the following URL for the training data file:
 https://introtomlsampledata.blob.core.windows.net/data/nyc-taxi/nyc-taxi-sample-data.csv
6.	Provide nyc-taxi-sample-data as the Name, leave the remaining values at their defaults and select Next.
 
Task 2: Preview Dataset
1.	On the Settings and preview panel, set the column headers drop down to All files have same headers.
2.	Scroll the data preview to right to observe the target column: totalAmount. After you are done reviewing the data, select Next
 
Task 3: Select Columns
1.	Select columns from the dataset to include as part of your training data. Leave the default selections and select Next
 
Task 4: Create Dataset
1.	Confirm the dataset details and select Create
 
Exercise 2: Create New Training Pipeline
Task 1: Open Pipeline Authoring Editor
1.	From the studio, select Designer, +. This will open a visual pipeline authoring editor.
 
Task 2: Setup Compute Target
1.	In the settings panel on the right, select Select compute target.
 
2.	In the Set up compute target editor, select the available compute, and then select Save.
Note: If you are facing difficulties in accessing pop-up windows or buttons in the user interface, please refer to the Help section in the lab environment.
 
Task 3: Add Dataset
1.	Select Datasets section in the left navigation. Next, select My Datasets, nyc-taxi-sample-data and drag and drop the selected dataset on to the canvas.
 
Task 4: Split Dataset
1.	Select Data Transformation section in the left navigation. Follow the steps outlined below:
1.	Select the Split Data prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Fraction of rows in the first output dataset: 0.7
4.	Connect the Dataset to the Split Data module
 
Note that you can submit the pipeline at any point to peek at the outputs and activities. Running pipeline also generates metadata that is available for downstream activities such selecting column names from a list in selection dialogs.
Task 5: Initialize Regression Model
1.	Select Machine Learning Algorithms section in the left navigation. Follow the steps outlined below:
1.	Select the Neural Network Regression prebuilt module, in the Regression category.
2.	Drag and drop the selected module on to the canvas
3.	Create trainer mode: Single Parameter. This option indicates how you want the model to be trained.
4.	Hidden layer specification: Fully connected case.
5.	For Learning rate: 0.01.
 
2.	Note: Because the number of nodes in the input layer is determined by the number of features in the training data, in a regression model there can be only one node in the output layer.
Task 6: Setup Train Model Module
1.	Select Model Training section in the left navigation. Follow the steps outlined below:
1.	Select the Train Model prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the Neural Network Regression module to the first input of the Train Model module
4.	Connect the first output of the Split Data module to the second input of the Train Model module
5.	Select the Edit column link to open the Label column editor
 
2.	The Label column editor allows you to specify your Label or Target column. Type in the label column name totalAmount and then select Save.
 
Task 7: Setup Score Model Module
1.	Select Model Scoring & Evaluation section in the left navigation. Follow the steps outlined below:
1.	Select the Score Model prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the Train Model module to the first input of the Score Model module
4.	Connect the second output of the Split Data module to the second input of the Score Model module
 
Note that Split Data module will feed data for both model training and model scoring. The first output (0.7 fraction) will connect with the Train Model module and the second output (0.3 fraction) will connect with the Score Model module.
Task 8: Setup Evaluate Model Module
1.	Select Model Scoring & Evaluation section in the left navigation. Follow the steps outlined below:
1.	Select the Evaluate Model prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the Score Model module to the first input of the Evaluate Model module
 
Exercise 3: Submit Training Pipeline
Task 1: Create Experiment and Submit Pipeline
1.	Select Submit to open the Setup pipeline run editor.
 
Please note that the button name in the UI is changed from Run to Submit.
2.	In the Setup pipeline run editor, select Experiment, Create new and provide New experiment name: neural-network-regression, and then select Submit.
 
3.	Wait for pipeline run to complete. It will take around 8 minutes to complete the run.
4.	While you wait for the model training to complete, you can learn more about the training algorithm used in this lab by selecting Neural Network Regression module.
Exercise 4: Visualize Training Results
Task 1: Visualize the Model Predictions
1.	Select Score Model, Outputs, Visualize to open the Score Model result visualization dialog or just simply right-click the Score Model module and select Visualize Scored Dataset.
 
2.	Observe the predicted values under the column Scored Labels. You can compare the predicted values (Scored Labels) with actual values (totalAmount).
 
Task 2: Visualize the Evaluation Results
1.	Select Evaluate Model, Outputs, Visualize to open the Evaluate Model result visualization dialog or just simply right-click the Evaluate Model module and select Visualize Evaluation Results.
 
2.	Evaluate the model performance by reviewing the various evaluation metrics, such as Mean Absolute Error, Root Mean Squared Error, etc.
 
Next Steps
Congratulations! You have trained a simple neural net model using the prebuilt Neural Network Regression module in the AML visual designer. You can continue to experiment in the environment but are free to close the lab environment tab and return to the Udacity portal to continue with the lesson.
5.8 Walkthrough: Train a Simple Neural Net
https://photos.app.goo.gl/y2nTRisMcEVe6yZg8
5.9 Specialized Cases of Model Training
In this section, we'll have a look at some of the specialized cases of model training. But first, let's review the main types of machine learning approaches that we introduced back in the first lesson.
A Review of Approaches to Machine Learning
As you now know, there are three main approaches to machine learning:
•	Supervised learning
•	Unsupervised learning
•	Reinforcement learning
https://photos.app.goo.gl/cBP3XjBqPwh8L8pu9

 


Specialized Cases of Model Training
https://photos.app.goo.gl/VjYzuGvPzvgzPMSj7
5.10 Prelaunch Lab
5.11 Similarity Learning
https://photos.app.goo.gl/s2VCzkharASL9wcD6

 
Recommender Systems
As we mentioned above, one of the most common uses of similarity learning is in creating recommender systems. Let's consider these in more detail.
https://photos.app.goo.gl/NAyW2VduvdAW5tmi6

 

5.12 Lab: Train a Simple Recommender
Train a simple recommender
The main aim of a recommendation system is to recommend one or more items to users of the system. Examples of an item to be recommended, might be a movie, restaurant, book, or song. In general, the user is an entity with item preferences such as a person, a group of persons, or any other type of entity you can imagine.
There are two principal approaches to recommender systems:
•	The content-based approach, which makes use of features for both users and items. Users can be described by properties such as age or gender. Items can be described by properties such as the author or the manufacturer. Typical examples of content-based recommendation systems can be found on social matchmaking sites.
•	The Collaborative filtering approach, which uses only identifiers of the users and the items. It is based on a matrix of ratings given by the users to the items. The main source of information about a user is the list the items they’ve rated and the similarity with other users who have rated the same items.
The SVD recommender module in Azure Machine Learning designer is based on the Singular Value Decomposition algorithm. It uses identifiers of the users and the items, and a matrix of ratings given by the users to the items. It’s a typical example of collaborative recommender.
Lab Overview
In this lab, we make use of the Train SVD Recommender module available in Azure Machine Learning designer (preview), to train a movie recommender engine. We use the collaborative filtering approach: the model learns from a collection of ratings made by users on a subset of a catalog of movies. Two open datasets available in Azure Machine Learning designer are used the IMDB Movie Titles dataset joined on the movie identifier with the Movie Ratings dataset. The Movie Ratings data consists of approximately 225,000 ratings for 15,742 movies by 26,770 users, extracted from Twitter using techniques described in the original paper by Dooms, De Pessemier and Martens. The paper and data can be found on GitHub.
We will both train the engine and score new data, to demonstrate the different modes in which a recommender can be used and evaluated. The trained model will predict what rating a user will give to unseen movies, so we’ll be able to recommend movies that the user is most likely to enjoy. We will do all of this from the Azure Machine Learning designer without writing a single line of code.
Exercise 1: Create New Training Pipeline
Task 1: Open Pipeline Authoring Editor
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	From the studio, select Designer, +. This will open a visual pipeline authoring editor.
 
Task 2: Setup Compute Target
1.	In the settings panel on the right, select Select compute target.
 
2.	In the Set up compute target editor, select the available compute, and then select Save.
Note: If you are facing difficulties in accessing pop-up windows or buttons in the user interface, please refer to the Help section in the lab environment.
 
Task 3: Add Sample Datasets
1.	Select Datasets section in the left navigation. Next, select Samples, Movie Ratings and drag and drop the selected dataset on to the canvas.
 
2.	Select Datasets section in the left navigation. Next, select Samples, IMDB Movie Titles and drag and drop the selected dataset on to the canvas.
 
Task 4: Join the two datasets on Movie ID
1.	Select Data Transformation section in the left navigation. Follow the steps outlined below:
1.	Select the Join Data prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the output of the Movie Ratings module to the first input of the Join Data module.
4.	Connect the output of the IMDB Movie Titles module to the second input of the Join Data module.
 
2.	Select the Join Data module.
3.	Select the Edit column link to open the Join key columns for left dataset editor. Select the MovieId column in the Enter column name field.
 
4.	Select the Edit column link to open the Join key columns for right dataset editor. Select the Movie ID column in the Enter column name field.
 
Note that you can submit the pipeline at any point to peek at the outputs and activities. Running pipeline also generates metadata that is available for downstream activities such selecting column names from a list in selection dialogs.
Task 5: Select Columns UserId, Movie Name, Rating using a Python script
1.	Select Python Language section in the left navigation. Follow the steps outlined below:
1.	Select the Execute Python Script prebuilt module.
2.	Drag and drop the selected module on to the canvas.
3.	Connect the Join Data output to the input of the Execute Python Script module.
 
2.	Select Edit code to open the Python script editor, clear the existing code and then enter the following lines of code to select the UserId, Movie Name, Rating columns from the joined dataset. Please ensure that there is no indentation for the first line and the second and third lines are indented.
3.	 def azureml_main(dataframe1 = None, dataframe2 = None):
4.	     df1 = dataframe1[['UserId','Movie Name','Rating']]
     return df1,
Note: In other pipelines, for selecting a list of columns from a dataset, we could have used the Select Columns from Dataset prebuilt module. This one returns the columns in the same order as in the input dataset. This time we need the output dataset to be in the format: user id, movie name, rating.This column order is required at the input of the Train SVD Recommender module.
Task 6: Remove duplicate rows with same Movie Name and UserId
1.	Select Data Transformation section in the left navigation. Follow the steps outlined below:
1.	Select the Remove Duplicate Rows prebuilt module.
2.	Drag and drop the selected module on to the canvas.
3.	Connect the first output of the Execute Python Script to the input of the Remove Duplicate Rows module.
4.	Select the Edit columns link to open the Select columns editor and then enter the following list of columns to be included in the output dataset: Movie Name, UserId.
 
Task 7: Split the dataset into training set (0.5) and test set (0.5)
1.	Select Data Transformation section in the left navigation. Follow the steps outlined below:
1.	Select the Split Data prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Fraction of rows in the first output dataset: 0.5
4.	Connect the Dataset to the Split Data module
 
Task 8: Initialize Recommendation Module
1.	Select Recommendation section in the left navigation. Follow the steps outlined below:
1.	Select the Train SVD Recommender prebuilt module.
2.	Drag and drop the selected module on to the canvas
3.	Connect the first output of the Split Data module to the input of the Train SVD Recommender module
4.	Number of factors: 200. This option specify the number of factors to use with the recommender. With the number of users and items increasing, it’s better to set a larger number of factors. But if the number is too large, performance might drop.
5.	Number of recommendation algorithm iterations: 30. This number indicates how many times the algorithm should process the input data. The higher this number is, the more accurate the predictions are. However, a higher number means slower training. The default value is 30.
6.	For Learning rate: 0.001. The learning rate defines the step size for learning.
 
Task 9: Select Columns UserId, Movie Name from the test set
1.	Select Data Transformation section in the left navigation. Follow the steps outlined below:
1.	Select the Select Columns in Dataset prebuilt module.
2.	Drag and drop the selected module on to the canvas.
3.	Connect the Split Data second output to the input of the Select columns in Dataset module.
4.	Select the Edit columns link to open the Select columns editor and then enter the following list of columns to be included in the output dataset: UserId, Movie Name.
 
Task 10: Configure the Score SVD Recommender
1.	Select Recommendation section in the left navigation. Follow the steps outlined below:
1.	Select the Score SVD Recommender prebuilt module.
2.	Drag and drop the selected module on to the canvas
3.	Connect the output of the Train SVD Recommender module to the first input of the Score SVD Recommender module, which is the Trained SVD Recommendation input.
4.	Connect the output of the Select Columns in Dataset module to the second input of the Score SVD Recommender module, which is the Dataset to score input.
5.	Select the Score SVD Recommender module on the canvas.
6.	Recommender prediction kind: Rating Prediction. For this option, no other parameters are required. When you predict ratings, the model calculates how a user will react to a particular item, given the training data. The input data for scoring must provide both a user and the item to rate.
 
Task 11: Setup Evaluate Recommender Module
1.	Select Recommendation section in the left navigation. Follow the steps outlined below:
1.	Select the Evaluate Recommender prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the Score SVD Recommender module to the second input of the Evaluate Recommender module, which is the Scored dataset input.
4.	Connect the second output of the Split Data module (train set) to the first input of the Evaluate Recommender module, which is the Test dataset input.
 
Exercise 2: Submit Training Pipeline
Task 1: Create Experiment and Submit Pipeline
1.	Select Submit on the right corner of the canvas to open the Setup pipeline run editor.
 
Please note that the button name in the UI is changed from Run to Submit.
2.	In the Setup pipeline run editor, select Experiment, Create new and provide New experiment name: movie-recommender, and then select Submit.
 
3.	Wait for pipeline run to complete. It will take around 20 minutes to complete the run.
4.	While you wait for the model training to complete, you can learn more about the SVD algorithm used in this lab by selecting Train SVD Recommender.
Exercise 3: Visualize Scoring Results
Task 1: Visualize the Scored dataset
1.	Select Score SVD Recommender, Outputs, Visualize to open the Score SVD Recommender result visualization dialog or just simply right-click the Score SVD Recommender module and select Visualize Scored dataset.
 
2.	Observe the predicted values under the column Rating.
 
Task 2: Visualize the Evaluation Results
1.	Select Evaluate Recommender, Outputs, Visualize to open the Evaluate Recommender result visualization dialog or just simply right-click the Evaluate Recommender module and select Visualize Evaluation Results.
 
2.	Evaluate the model performance by reviewing the various evaluation metrics, such as Mean Absolute Error, Root Mean Squared Error, etc.
 
Next Steps
Congratulations! You have trained a simple movie recommender using the prebuilt Recommender modules in the AML visual designer. You can continue to experiment in the environment but are free to close the lab environment tab and return to the Udacity portal to continue with the lesson.
5.13 Walkthrough: Train a Simple Recommender
https://photos.app.goo.gl/JsrWKe9fPVhjkSwBA
https://photos.app.goo.gl/NzmWaiJ1g1WZVCBeA
5.14 Prelaunch Lab
5.15 Text Classification
https://photos.app.goo.gl/96yB7xSjEFAmQAvS6
Remember that before we can do text classification, the text first needs to be translated into some kind of numerical representation—a process known as text embedding. The resulting numerical representation, which is usually in the form of vectors, can then be used as an input to a wide range of classification algorithms.
Training a Classification Model with Text
https://photos.app.goo.gl/pUVcR6D4chKJ2z1m7
As we mentioned in the video, an important part of the pipeline is the process of vectorizing the text, using a technique such as Term Frequency-Inverse Document Frequency (TF-IDF) vectorization. We discussed text vectorization in some detail back in the introduction lesson when we described text data—so you may want to go back to that section if you feel like you need a review of the concept.
https://photos.app.goo.gl/RUJWpScYo4kFt3Xq9


 


5.16 Lab: Train a Simple Text Classifier
Train a simple text classifier
In text classification scenarios, the goal is to assign a piece of text, such as a document, a news article, a search query, an email, a tweet, support tickets, customer feedback, user product review, to predefined classes or categories. Some examples of text classification applications are: categorizing newspaper articles into topics, organizing web pages into hierarchical categories, spam email filtering, sentiment analysis, predicting user intent from search queries, support tickets routing, and customer feedback analysis.
Lab Overview
In this lab we demonstrate how to use text analytics modules available in Azure Machine Learning designer (preview) to build a simple text classification pipeline. We will create a training pipeline and initialize a multiclass logistic regression classifier to predict the company category with Wikipedia SP 500 dataset derived from Wikipedia. The dataset manages articles of each S&P 500 company. Before uploading to Azure Machine Learning designer, the dataset was processed as follows: extracted text content for each specific company, removed wiki formatting, removed non-alphanumeric characters, converted all text to lowercase, known company categories added. Articles could not be found for some companies, so that’s why the number of records is less than 500.
Exercise 1: Create New Training Pipeline
Task 1: Open Pipeline Authoring Editor
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	From the studio, select Designer, +. This will open a visual pipeline authoring editor.
 
Task 2: Setup Compute Target
1.	In the settings panel on the right, select Select compute target.
 
2.	In the Set up compute target editor, select the available compute, and then select Save.
Note: If you are facing difficulties in accessing pop-up windows or buttons in the user interface, please refer to the Help section in the lab environment.
 
Task 3: Add Wikipedia SP 500 Sample Datasets
1.	Select Datasets section in the left navigation. Next, select Samples, Wikipedia SP 500 Dataset and drag and drop the selected dataset on to the canvas.
 
Task 4: Preprocess text for following steps
1.	Select Text Analytics section in the Modules . Follow the steps outlined below:
1.	Select the Preprocess Text prebuilt module.
2.	Drag and drop the selected module on to the canvas.
3.	Connect the output of the Wikipedia SP 500 module to the input of the Preprocess Text module. The dataset input of this prebuilt module needs to be connected to a dataset that has at least one column containing text.
4.	Select the language from the Language dropdown list: English.
5.	Select the Edit column link to open the Text column to clean editor. Select the Text column in the Enter column name field.
6.	Leave all the other options checked, as in the default configuration of the Preprocess Text module.
 
Task 5: Split the dataset into training set (0.5) and test set (0.5)
1.	Select Data Transformation section in the Modules. Follow the steps outlined below:
1.	Select the Split Data prebuilt module.
2.	Drag and drop the selected module on to the canvas.
3.	Fraction of rows in the first output dataset: 0.5.
4.	Stratified seed: True.
5.	Select the Edit column link to open the Stratification key column editor. Select the Category column in the Enter column name field.
6.	Connect the output of the Preprocess Text module to the input of the Split Data module.
 
Task 6: Convert the plain text of the articles to integers with Feature Hashing module, on the training set
1.	Select Text Analytics section in the left navigation. Follow the steps outlined below:
1.	Select the Feature Hashing prebuilt module.
2.	Drag and drop the selected module on to the canvas.
3.	Connect the first output of the Split Data to the input of the Feature Hashing module.
4.	Select the Edit columns link to open the Target column editor and then enter the Preprocessed Text column for the Target column field.
5.	Set Hashing bitsize: 10 and set the number of N-grams: 2.
 
The goal of using feature hashing is to reduce dimensionality; also it makes the lookup of feature weights faster at classification time because it uses hash value comparison instead of string comparison.
Task 7: Featurize unstructured text data with Extract N-Gram Feature from Text module, on the training set
1.	Select Text Analytics section in the left navigation. Follow the steps outlined below:
1.	Select the Extract N-Gram Feature from Text prebuilt module.
2.	Drag and drop the selected module on to the canvas.
3.	Connect the first output of the Split Data to the input of the Extract N-Gram Feature from Text module.
4.	Select the Edit columns link to open the Text column editor and then enter the Preprocessed Text column for the Target column field.
5.	Leave default selection of Vocabulary mode to Create to indicate that you’re creating a new list of n-gram features.
6.	Set N-grams size: 2 (which is the maximum size of the n-grams to extract and store)
7.	Select Weighting function: TF-IDF Weight. (This function calculates a term frequency/inverse document frequency score and assigns it to each n-gram. The value for each n-gram is its TF score multiplied by its IDF score.)
8.	Check the option to Normalize n-gram feature vectors. If this option is enabled, each n-gram feature vector is divided by its L2 norm.
 
Task 8: Remove text columns from dataset
1.	Select Data Transformation section in the left navigation. Follow the steps outlined below to add two Select Columns in Dataset modules on both featurization branches:
1.	Select the Select Columns in Dataset prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the input Select Columns in Dataset module to the Feature Hashing module output.
 
2.	Select the Edit columns link to open the Select columns editor and then select All columns.
3.	Click on the + icon to add another operation line and select the Exclude, Column names option. Enter to following list of columns to be excluded: Title, Text, Preprocessed Text. Select Save to close the editor.
 
4.	Select the Select Columns in Dataset module and click on the Copy icon and then Paste it on the canvas. Connect the copied module to the Extract N-Gram Feature from Text module.
 
Task 9: Add the Train Model Modules
1.	Select Model Training section in the left navigation. Follow the steps outlined below:
1.	Select the Train Model prebuilt module.
2.	Drag and drop the selected module on to the canvas
3.	Connect the first output of the Select Columns in Dataset module on the left of the canvas, to the second input (the Dataset input) of the newly added Train Model module.
4.	Select the Edit columns link to open the Label column editor and then enter the Category column. Select Save to close the editor.
5.	Select the Train Model module and click on the Copy icon.
6.	Click on the Paste icon to paste a second Train Model module on the canvas under the right branch of the pipeline tree.
7.	Connect this last Train Model module, by linking at the second input (the Dataset input), the output of the right most Select Columns in Dataset module on the canvas.
 
Task 10: Initialize Multiclass Logistic Regression Model
1.	Select Machine Learning Algorithms section in the left navigation. Follow the steps outlined below:
1.	Select the Multiclass Logistic Regression prebuilt module, in the Classification category.
2.	Drag and drop the selected module on to the canvas
3.	Connect the output of the Multiclass Logistic Regression module to the first input of the left branch (the Feature Hashing approach) Train Model module.
4.	Connect the output of the Multiclass Logistic Regression module to the first input of the right branch (the N-Gram Features approach) Train Model module.
 
Task 11: Convert the plain text of the articles to integers with Feature Hashing module, on the test set
1.	Select the existing Feature Hashing module.
2.	Copy the Feature Hashing module using the Copy option from the top menu.
3.	Paste the copied module on the canvas.
4.	Position the new Feature Hashing module between the two Train Model modules on the canvas.
5.	Connect the second output of the Split Data, the test set output, to the input of the newly added Feature Hashing module.
 
Task 12: Featurize unstructured text data with Extract N-Gram Feature from Text module, on the test set
1.	Select the existing Extract N-Gram Feature from Text module.
2.	Copy the Extract N-Gram Feature from Text module using the Copy option from the top menu.
3.	Paste the copied module on the canvas.
4.	Position the new Extract N-Gram Feature from Text module near the rightmost Train Model module on the canvas.
5.	Connect the second output of the Split Data, the test set output, to the first input of the newly added Extract N-Gram Feature from Text module.
6.	Connect the second output of the uppermost Extract N-Gram Feature from Text, to the second input of the copied Extract N-Gram Feature from Text module.
7.	Select the newly added module and in the right settings pane, set the Vocabulary mode to ReadOnly
 
Task 13: Setup Score Model Modules
1.	Select Model Scoring & Evaluation section in the left navigation. Follow the steps outlined below:
1.	Select the Score Model prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the leftmost Train Model module to the first input of the Score Model module
4.	Connect the output of the Feature Hashing module (the lower one, on the test set branch) to the second input of the Score Model module
 
2.	Select Model Scoring & Evaluation section in the left navigation. Follow the steps outlined below:
1.	Select the Score Model prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the rightmost Train Model module to the first input of the Score Model module
4.	Connect the first output of the Extract N-Gram Feature from Text module (the lower one, on the test set branch) to the second input of the rightmost Score Model module.
 
Task 14: Setup Evaluate Model Module
1.	Select Model Scoring & Evaluation section in the left navigation. Follow the steps outlined below:
1.	Select the Evaluate Model prebuilt module
2.	Drag and drop the selected module on to the canvas
3.	Connect the two Score Model modules to the inputs of the Evaluate Model module
 
Exercise 2: Submit Training Pipeline
Task 1: Create Experiment and Submit Pipeline
1.	Select Submit to open the Setup pipeline run editor.
 
Please note that the button name in the UI is changed from Run to Submit.
2.	In the Setup pipeline run editor, select Experiment, Create new and provide New experiment name: wiki-text-classifier, and then select Submit.
 
3.	Wait for pipeline run to complete. It will take around 20 minutes to complete the run.
4.	While you wait for the model training to complete, you can learn more about the classification algorithm used in this lab by selecting Multiclass Logistic Regression module.
Exercise 3: Visualize Training Results
Task 1: Visualize the Model Predictions
1.	Select Score Model, Outputs, Visualize to open the Score Model result visualization dialog or just simply right-click the Score Model module and select Visualize Scored Dataset.
 
2.	Observe the predicted values under the column Category.
 
Task 2: Visualize the Evaluation Results
1.	Select Evaluate Model, Outputs, Visualize to open the Evaluate Model result visualization dialog or just simply right-click the Evaluate Model module and select Visualize Evaluation Results.
 
2.	Evaluate the model performance by reviewing the various evaluation metrics. Evaluate Model has two input ports, so that we could evaluate and compare scored datasets that are generated with different methods. In this sample, we compare the performance of the result generated with feature hashing method and n-gram method.
 
Next Steps
Congratulations! You have trained a simple text classifier and compared performance of the result generated with two different featurization modules. You can continue to experiment in the environment but are free to close the lab environment tab and return to the Udacity portal to continue with the lesson.
5.17 Walkthrough: Train a Simple Text Classifier
https://photos.app.goo.gl/K7chNZ9dQPB1P9jT7
https://photos.app.goo.gl/WEmkem114QeuiQcD6
5.18 Feature Learning
As we've discussed previously, feature engineering is one of the core techniques that can be used to increase the chances of success in solving machine learning problems. As a part of feature engineering, feature learning (also called representation learning) is a technique that you can use to derive new features in your dataset. Let's have a look at how this technique works.
Supervised and Unsupervised Approaches
Earlier in this lesson, we pointed out that feature learning is one of the machine learning techniques that can be done in both supervised and unsupervised ways. Let's have a look at both approaches.
https://photos.app.goo.gl/ZQp323UD1wCB8dna6

 

5.19 Applications of Feature Learning
https://photos.app.goo.gl/4sCwFQUti5edsj1u8
As we just mentioned, some prominent applications of machine learning include image classification and image search. On the remainder of this page, we'll take a closer look at some specific examples of these two approaches.
Image Classification with Convolutional Neural Networks (CNNs)
https://photos.app.goo.gl/TnwDfnR6iNnus8EY6
Image Search with Autoencoders
https://photos.app.goo.gl/Nbxkqxzv65p66W8U9
5.20 Anomaly Detection
Datasets often contain a small number of items that deviate significantly from the norm. These anomalies can be of interest, since they may be the result of bad data, unusual behavior, or important exceptions to the typical trends. Anomaly detection is a machine learning technique concerned with finding these data points.
Supervised and Unsupervised Approaches
Anomaly detection is another one of the machine learning techniques that can be done in both supervised and unsupervised ways. Let's have a look at both approaches.
https://photos.app.goo.gl/RqwEFV1ARGgMNiB29

 

Applications of Anomaly Detection
https://photos.app.goo.gl/7SBNVj6i2PQtLUjc6
Let's now look at a specific example of anomaly detection, so that we can get a more concrete idea of what the process might look like. In this particular example, we'll consider what anomaly detection might look like when applied to machinery maintenance.
https://photos.app.goo.gl/oo8j3EAbFfRenSzz5

 

5.21 Prelaunch Lab
5.22 Forecasting
Now let's have a look at the problem of forecasting. A typical example of a forecasting problem would be: Given a set of ordered data points (such as sales data over a series of dates), we want to predict the next data points in the series (such as what sales will look like next week).
https://photos.app.goo.gl/UG2pR7bFk4ez9UuJ7
Remember, forecasting is a class of problems that deals with predictions in the context of orderable datasets. These orderable datasets can be time-series datasets, but they don't have to be—forecasting can be applied to other types of orderable sets as well.
Types of Forecasting Algorithms
https://photos.app.goo.gl/gxyb1DoMxkzfhjUk9

 

5.23 Lab: Forecasting
Train a time-series forecasting model using Automated Machine Learning
Lab Overview
In this lab you will learn how the Automated Machine Learning capability in Azure Machine Learning (AML) can be used for the life cycle management of the manufactured vehicles and how AML helps in creation of better vehicle maintenance plans. To accomplish this, you will train a Linear Regression model to predict the number of days until battery failure using Automated Machine Learning available in AML studio.
Exercise 1: Creating a model using automated machine learning
Task 1: Create an automated machine learning experiment using the Portal
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	Select Automated ML in the left navigation bar.
 
5.	Select New automated ML run to start creating a new experiment.
 
6.	Select Create dataset and choose the From web files option from the drop-down.
 
7.	Fill in the training data URL in the Web URL field: https://introtomlsampledata.blob.core.windows.net/data/battery-lifetime/training-formatted.csv, make sure the name is set to training-formatted-dataset, and select Next to load a preview of the parsed training data.
 
8.	In the Settings and preview page, for the Column headers field, select All files have same headers. Scroll to the right to observe all of the columns in the data.
 
9.	Select Next to check the schema and then confirm the dataset details by selecting Next and then Create on the confirmation page.
 
10.	Now you should be able to select the newly created dataset for your experiment. Select the training-formatted-dataset dataset and select Next to move to the experiment run details page.
 
11.	You will now configure the Auto ML run basic settings by providing the following values for the experiment name, target column and training compute:
o	Experiment name: automlregression
o	Target column: select Survival_In_Days
o	Select training compute target: : select qs-compute
 
12.	Select Next and select Regression in the Task type and settings page.
 
13.	Select View additional configuration settings to open the advanced settings section. Provide the following settings:
o	Primary metric: Normalized root mean squared error
o	Exit criterion > Metric score threshold: 0.09
o	Validation > Validation type: k-fold cross validation
o	Validation > Number of Cross Validations: 5
o	Concurrency > Max concurrent iterations: 1
 
14.	Select Save and then Finish to begin the automated machine learning process.
 
15.	Wait until the Run status becomes Running in the Run Detail page.
 
Task 2: Review the experiment run results
1.	The experiment will run for about 15 minutes. While it runs and once it completes, you should check the Models tab on the Run Detail page to observe the model performance for the primary metric for different runs.
 
2.	In the models list, notice at the top the iteration with the best normalized root mean square error score. Note that the normalized root mean square error measures the error between the predicted value and actual value. In this case, the model with the lowest normalized root mean square error is the best model.
 
3.	Select Experiments on the left navigation pane and select the experiment automlregression to see the list of available runs.
 
4.	Select the option to Include child runs to be able to examine model performance for the primary metric of different runs. By default, the left chart describes the normalized_median_absolute_error value for each run. Select the pen icon on the right corner of the normalized_median_absolute_error chart to configure the normalized_root_mean_square_error metric representation.
 
Next Steps
Congratulations! You have trained a simple time-series forecasting model using automated machine learning in the visual interface. You can continue to experiment in the environment but are free to close the lab environment tab and return to the Udacity portal to continue with the lesson.
5.24 Walkthrough: Forecasting
https://photos.app.goo.gl/YTutXZefTzJFcm2m7
5.25 Lesson Summary
https://photos.app.goo.gl/v94N6pirF3ZKpz4G9
In this lesson, you've learned the fundamentals of deep learning, including:
•	The differences between classical machine learning and deep learning
•	The benefits and applications of Deep Learning
•	How to train your first neural network model
Next, you learned about some of the most important specialized cases of model training, including:
•	Similarity learning and the basic features of a recommendation engine
•	Text classification and the fundamentals of processing text in machine learning
•	Feature learning, an essential task in feature engineering
•	Anomaly detection
•	Time-series forecasting.
Along the way, you got practice with several hands-on labs, using the Designer in Azure Machine Learning Studio to train a simple neural network, a recommendation engine, a text classifier, and a time-series forecasting model.
6 Managed Services For Machine Learning
6.1 Lesson Overview
https://photos.app.goo.gl/ghX1dXhu6yKUNxrJ8
This lesson covers managed services for Machine Learning, which are services you use to enhance your Machine Learning processes. We will use services provided by Azure Machine Learning as examples throughout the lesson.
You will learn about various types of computing resources made available through managed services, including:
•	Training compute
•	Inferencing compute
•	Notebook environments
You will also study the main concepts involved in the modeling process, including:
•	Basic modeling
•	How parts of the modeling process interact when used together
•	More advanced aspects of the modeling process, like automation via pipelines and end-to-end integrated processes (also known as DevOps for Machine Learning or simply, MLOps)
•	How to move the results of your modeling work to production environments and make them operational
Finally, you will be introduced to the world of programming the managed services via the Azure Machine Learning SDK for Python.
6.2 Managed Services for Machine Learning
https://photos.app.goo.gl/HrRJCzpXLG19s4Km6
The machine learning process can be labor intensive. Machine learning requires a number of tools to prepare the data, train the models, and deploy the models. Most of the work usually takes place within web-based, interactive notebooks, such as Jupyter notebooks. Although notebooks are lightweight and easily run in a web browser, you still need a server to to host them. Typically, this involves installing several applications and libraries on a machine, configuring the environment settings, and then loading any additional resources required to begin working within notebooks or integrated development environments (IDEs).
All this setup takes time, and there is sometimes a fair amount of troubleshooting involved to make sure you have the right combination of software versions that are compatible with one another. This is the advantage of managed services for machine learning, which provide a ready-made environment that is pre-optimized for your machine learning development.
6.3 Prelaunch Lab
6.4 Compute Resources
A compute target is a designated compute resource or environment where you run training scripts or host your service deployment. There are two different variations on compute targets that we will discuss below: training compute targets and inferencing compute targets.
Training Compute
First, let's talk about compute resources that can be used for model training.
https://photos.app.goo.gl/XHs1SfmW4yZZ1sk96
Inferencing Compute
Once you have a trained model, you'll want to be able to deploy it for inferencing. Let's take a look at the compute resources you can use for different types of inferencing.
https://photos.app.goo.gl/QQtfdJP5E1sfV5HW6

 
6.5 Lab: Managing Compute
Managing a compute instance
Machine learning requires several tools to prepare data, and train and deploy models. Most of the work usually takes place within web-based, interactive notebooks, such as Jupyter notebooks. Although notebooks are lightweight and easily run in a web browser, you still need a server to to host them.
So, the setup process for most users is to install several applications and libraries on a machine, configure the environment settings, then loaed any additional resources to begin working within notebooks or integrated development environments (IDEs). All this setup takes time, and there is sometimes a fair amount of troubleshooting involved to make sure you have the right combination of software versions that are compatible with one another.
What if you could use a ready-made environment that is pre-optimized for your machine learning development?
Azure Machine Learning compute instance provides this type of environment for you, and is fully managed, meaning you don’t have to worry about setup and applying patches and updates to the underlying virtual machine. Plus, since it is cloud-based, you can run it from anywhere and from any machine. All you need to do is specify the type of virtual machine, including GPUs and I/O-optimized options, then you have what you need to start working.
The managed services, such as computer instance and compute cluster, can be used as a training compute target to scale out training resources to handle larger data sets. When you are ready to run your experiments and build your models, you need to specify a compute target. Compute targets are compute resources where you run your experiments or host your service deployment. The target may be your local machine or a cloud-based resource. This is another example of where managed services like compute instance and computer cluster really shine.
A managed compute resource is created and managed by Azure Machine Learning. This compute is optimized for machine learning workloads.
Azure Machine Learning compute clusters and compute instances are the only managed computes. Additional managed compute resources may be added in the future.
Overview
In this lab, you will explore different actions you can take to manage a compute instance in Azure Machine Learning Studio.
Exercise 1: Create New Compute Instance
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	From the studio, navigate to Compute, then select +New.
 
5.	In the New Compute Instance pane, provide the following information and then select Create.
o	Compute name: provide an unique name
o	Virtual Machine size: Standard_D3_v2
 
6.	It will take couple of minutes for your compute instance to be ready. Wait for your compute instance to be in status Running.
Exercise 2: Explore Compute Instances
1.	Select the radio button next to the name of your compute instance. This will select the instance, as indicated by a checkmark. Selecting your instance in this way enables the toolbar options above that enable you to Stop, Restart, or Delete the instance.
 
There are different scenarios in which you will want to perform these actions. Here are the actions you can take on a selected compute instance, and what they do:
o	Stop: Since the compute instance runs on a virtual machine (VM), you pay for the instance as long as it is running. Naturally, it needs to run to perform compute tasks, but when you are done using it, be sure to stop it with this option to prevent unnecessary costs.
o	Restart: Restarting an instance is sometimes necessary after installing certain libraries or extensions. There may be times, however, when the compute instance stops functioning as expected. When this happens, try restarting it before taking further action.
o	Delete: You can create and delete instances as you see fit. The good news is, all notebooks and R scripts are stored in the default storage account of your workspace in Azure file share, within the “User files” directory. This central storage allows all compute instances in the same workspace to access the same files so you don’t lose them when you delete an instance you no longer need.
2.	Select the name of your instance. This opens the Compute details blade, revealing useful information about your compute instance.
 
The Attributes describe the resource details of the compute instance, including the name, type, Azure subscription, the resource group to which it belongs, the Azure Machine Learning workspace that manages it, and the Azure region to which it is deployed. If you need to execute scripts that require details about your compute instance, this is where you can find most of what you need.
The Resource properties show the status and configuration of the compute instance, including links to its applications and public and private endpoints. In this screenshot, you will see that SSH access is disabled. You cannot enable SSH access after creating a compute instance. You can only enable this option at the time of creation. SSH access allows you to securely connect to the VM from a terminal or command window. Use the public IP address to connect via SSH or an integrated development environment (IDE) like Visual Studio Code.
3.	Navigate back to Compute. The compute instance comes preconfigured with tools and environments that enable you to author, train, and deploy models in a fully integrated notebook experience. You access these environments through the Application URI links located in the resource properties (as seen in the previous step), and next to each compute instance in the list.
 
4.	Select each of the application links to sign in to the related environment. You may be prompted to select your user account for each application.
Next Steps
Congratulations! You have completed the introduction to managing a compute instance lab. You can continue to experiment in the environment but are free to close the lab environment tab and return to the Udacity portal to continue with the lesson.

6.6 Walkthrough: Managing Compute
https://photos.app.goo.gl/jn7ZV6HCVKuf63R7A

6.7 Prelaunch Lab
6.8 Managed Notebook Environments
https://photos.app.goo.gl/t8BiVC1wGd82C98a8
Notebooks are made up of one or more cells that allow for the execution of the code snippets or commands within those cells. They store commands and the results of running those commands. In this diagram, you can see that we can use a notebook environment to perform the five primary stages of model development:
 
The best way to understand how managed notebook environments work is to jump in and work with one—so that's what we'll do in the next lab, by training a simple SciKit learn model on a medical dataset.

6.9 Lab: Managed Notebook Environments
Compute Resources

Train a machine learning model from a managed notebook environment
So far, the Managed Services for Azure Machine Learning lesson has covered compute instance and the benefits it provides through its fully managed environment containing everythng you need to run Azure Machine Learning. Now it is time to gain some hands-on experience by putting a compute instance to work.
Overview
In this lab, you learn the foundational design patterns in Azure Machine Learning, and train a simple scikit-learn model based on the diabetes data set. After completing this lab, you will have the practical knowledge of the SDK to scale up to developing more-complex experiments and workflows.

In this tutorial, you learn the following tasks:
•	Connect your workspace and create an experiment
•	Load data and train a scikit-learn model
Exercise 1: Run the Notebook for this Lab
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	From the studio, navigate to Compute. Next, for the available Compute Instance, under Application URI select Jupyter. Be sure to select Jupyter and not JupterLab.
 
5.	From within the Jupyter interface, select New, Terminal.
 
6.	In the new terminal window run the following command and wait for it to finish:
git clone https://github.com/solliancenet/udacity-intro-to-ml-labs.git
 
7.	From within the Jupyter interface, navigate to directory udacity-intro-to-ml-labs/aml-visual-interface/lab-19/notebook and open 1st-experiment-sdk-train-model.ipynb. This is the Python notebook you will step through executing in this lab.
 
8.	Follow the instructions within the notebook to complete the lab.
9.	After completing the notebook, navigate back to the Notebook folder, then select the newly created outputs sub-folder. Here you will see the trained models (*.pkl files) generated by the last cell you executed. In addition, the serialized model is uploaded to each run. This allows you to download the model file from the run in the portal as an alternative to downloading them from this folder.
 
Next Steps
Congratulations! You have just learned how to use the Jupyter application on a compute instance to train a model. You can now return to the Udacity portal to continue with the lesson.
6.10 Walkthrough: Managed Notebook Environments
https://photos.app.goo.gl/3FykWdAsobgNLUMH9

6.11 Prelaunch Lab

6.12 Basic Modeling
Training, evaluating, and selecting the right Machine Learning models is at the core of each modern data science process. But what concrete steps do we need to go through to produce a trained model? In this section, we'll look at some important parts of the process and how we can use Azure Machine Learning to carry them out.
https://photos.app.goo.gl/uTN5B8TvXR291zdv8
Experiments
Before you create a new run, you must first create an experiment. Remember, an experiment is a generic context for handling runs. Think about it as a logical entity you can use to organize your model training processes.
https://photos.app.goo.gl/3q59fsgfYAHhZqte9
Runs
Once you have an experiment, you can create runs within that experiment. As we discussed above, model training runs are what you use to build the trained model. A run contains all artifacts associated with the training process, like output files, metrics, logs, and a snapshot of the directory that contains your scripts.
https://photos.app.goo.gl/KeYeFSA7jRQLRu5n7
Models
A run is used to produce a model. Essentially, a model is a piece of code that takes an input and produces output. To get a model, we start with a more general algorithm. By combining this algorithm with the training data—as well as by tuning the hyperparameters—we produce a more specific function that is optimized for the particular task we need to do. Put concisely:
Model = algorithm + data + hyperparameters
https://photos.app.goo.gl/LovCdBhmMJtbG4YL9


Model Registry
Once we have a trained model, we can turn to the model registry, which keeps track of all models in an Azure Machine Learning workspace. Note that models are either produced by a Run or originate from outside of Azure Machine Learning (and are made available via model registration).
https://photos.app.goo.gl/J7AdXLMTmxkbp6MZ9

 
6.13 Lab: Explore Experiments and Runs
Compute Resources
Explore experiments and runs
In the previous lab (19), you executed a Jupyter notebook that trained a model through a series of 10 different runs, each with a different alpha hyperparameter applied. These runs were created within the experiment you created at the beginning of the notebook. Because of this, Azure Machine Learning logged the details so you can review the result of each run and see how the alpha value is different between the them.
Overview
In this lab, you view the experiments and runs executed by a notebook. In the first part of the lab, you will use a notebook to create and run the experiments. In the second part of the lab, you will navigate to the Experiments blade in Azure Machine Learning Studio. Here you see all the individual runs in the experiment. Any custom-logged values (alpha_value and rmse, in this case) become fields for each run, and also become available for the charts and tiles at the top of the experiment page. To add a logged metric to a chart or tile, hover over it, click the edit button, and find your custom-logged metric.
When training models at scale over hundreds and thousands of separate runs, this page makes it easy to see every model you trained, specifically how they were trained, and how your unique metrics have changed over time.
Exercise 1: Run the Notebook for this Lab
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	From the studio, navigate to Compute. Next, for the available Compute Instance, under Application URI select Jupyter. Be sure to select Jupyter and not JupterLab.
 
5.	From within the Jupyter interface, select New, Terminal.
 
6.	In the new terminal window run the following command and wait for it to finish:
git clone https://github.com/solliancenet/udacity-intro-to-ml-labs.git
 
7.	From within the Jupyter interface, navigate to directory udacity-intro-to-ml-labs/aml-visual-interface/lab-20/notebook and open 1st-experiment-sdk-train-model.ipynb. This is the Python notebook you will step through executing in this lab.
 
8.	Follow the instructions within the notebook to complete the exercise.
Exercise 2: Open Experiments in the portal
1.	Within Azure Machine Learning Studio, select Experiments in the left-hand menu, then select the diabetes-experiment submitted by the notebook you executed in the previous lab (19).
 
2.	Here you can view details about the experiment and each of its runs, which created a new version of the model.
 
3.	Select Edit table in the top toolbar. In the Edit table dialog that appears, add the End time and Start time columns to the Selected columns list, then select Save.
 
Depending on your screen resolution, you might need to scroll down the table to see the bottom horizontal scrollbar. When you scroll all the way to the right, you will see the new columns you added.
 
4.	Select either the Run number or the Run ID of one of the runs to view its details. Both links on a run display the same dialog.
 
5.	The Details tab shows you more detailed information about each run, including the run time and metrics.
 
6.	Select the Outputs + logs tab. You see the .pkl file for the model that was uploaded to the run during each training iteration. This lets you download the model file rather than having to retrain it manually.
 
Next Steps
Congratulations! You have just learned how to use the Azure Machine Learning SDK to help you explain what influences the predictions a model makes. You can now return to the Udacity portal to continue with the lesson.

6.14 Walkthrough: Explore Experiments and Runs
https://photos.app.goo.gl/vBPjJp5gYPfSwyTq8
6.15 Advanced Modeling
https://photos.app.goo.gl/Y8nrXNjbnhnUJBBNA
Machine Learning Pipelines
As the process of building your models becomes more complex, it becomes more important to get a handle on the steps to prepare your data and train your models in an organized way. In these scenarios, there can be many steps involved in the end-to-end process, including:
•	Data ingestion
•	Data preparation
•	Model building & training
•	Model deployment.
These steps are organized into machine learning pipelines.
https://photos.app.goo.gl/NjwDGknjdobF49Yo7
MLOps: Creating Automatic End-to-End Integrated Processes
As we said earlier, we don't want all the steps in the machine learning pipeline to be manual—rather, we want to develop processes that use automated builds and deployments. The general term for this approach is DevOps; when applied to machine learning, we refer to the automation of machine learning pipelines as MLOps.
https://photos.app.goo.gl/5Qy9ao1NdZ2Gsj9f6
6.16 Prelaunch Lab
6.17 Operationalizing Models
After you have trained your machine learning model and evaluated it to the point where you are ready to use it outside your own development or test environment, you need to deploy it somewhere. Another term for this is operationalization.
https://photos.app.goo.gl/SanKjHB7mUeCejGy8
Real-time Inferencing
The model training process can be very compute-intensive, with training times that can potentially spann across many hours, days, or even weeks. A trained model, on the other hand, is used to make decisions on new data quickly. In other words, it infers things about new data it is given based on its training. Making these decisions on new data on-demand is called real-time inferencing.
https://photos.app.goo.gl/VFHS4ojZhJD7rBvL6



Batch Inferencing
Unlike real-time inferencing, which makes predictions on data as it is received, batch inferencing is run on large quantities (batches) of existing data. Typically, batch inferencing is run on a recurring schedule against data stored in a database or other data store.
https://photos.app.goo.gl/pCpbQqGqySdmzdbM6

 


6.18 Lab: Deploy a Model as a Webservice
Compute Resources

Deploy a trained model as a webservice

In previous lessons, we spent much time talking about training a machine learning model, which is a multi-step process involving data preparation, feature engineering, training, evaluation, and model selection. The model training process can be very compute-intensive, with training times spanning across many hours, days, or weeks depending on the amount of data, type of algorithm used, and other factors. A trained model, on the other hand, is used to make decisions on new data quickly. In other words, it infers things about new data it is given based on its training. Making these decisions on new data on-demand is called real-time inferencing.

Overview

In this lab, you learn how to deploy a trained model that can be used as a webservice, hosted on an Azure Kubernetes Service (AKS) cluster. This process is what enables you to use your model for real-time inferencing.
The Azure Machine Learning designer simplifies the process by enabling you to train and deploy your model without writing any code.
Exercise 1: Open a sample training pipeline
Task 1: Open the pipeline authoring editor
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.


 

3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:


 

For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.

4.	From the studio, select Designer in the left-hand menu. Next, select Sample 1: Regression - Automobile Price Prediction (Basic) under the New pipeline section. This will open a visual pipeline authoring editor.



 

Task 2: Setup the compute target

1.	In the settings panel on the right, select Select compute target.

 


2.	In the Set up compute target editor, select the existing compute target, then select Save.
Note: If you are facing difficulties in accessing pop-up windows or buttons in the user interface, please refer to the Help section in the lab environment.
 

Task 3: Create a new experiment and submit the pipeline
1.	Select Submit to open the Set up pipline run editor.

 

Please note that the button name in the UI is changed from Run to Submit.
2.	In the Setup pipeline run editor, select Experiment, Create new and provide an unique Experiment Name, and then select Submit.


 


3.	Wait for the pipeline run to complete. It will take around 10 minutes to complete the run.
Exercise 2: Real-time inference pipeline
Task 1: Create pipeline
1.	Select Create inference pipeline, then select Real-time inference pipeline from the list to create a new inference pipeline.


 

Task 2: Submit the pipeline
Select Submit to open the Set up pipeline run editor.

 

Please note that the button name in the UI is changed from Run to Submit.
1.	In the Setup pipeline run editor, select Select existing, then select the experiment you created in an earlier step. Select Submit to start the pipeline.


 


2.	Wait for pipeline run to complete. It will take around 7 minutes to complete the run.

Exercise 3: Deploy web service on Azure Kubernetes Service compute
Task 1: Deploy the web service
1.	After the inference pipeline run is finished, select Deploy to open the Set up real-time endpoint editor.


 


2.	In the Set up real-time endpoint editor, select your existing compute target, then select Deploy.



 


3.	Wait for the deployment to complete. The status of the deployment can be observed above the Pipeline Authoring Editor.


 

Task 2: Review deployed web service
1.	To view the deployed web service, select the Endpoints section in your Azure Portal Workspace.
2.	Select the deployed web service: sample-1-regression—-automobile to open the deployment details page.


 

Note: you have to select the text of the service name to open the deployment details page
Task 3: Review how to consume the deployed web service
1.	Select the Consume tab to observe the following information:
1.	Basic consumption info displays the REST endpoint, Primary key, and Secondary key.
2.	Consumption option shows code samples in C#, Python, and R on how to call the endpoint to consume the webservice.



 

Next Steps
Congratulations! You have just learned how to train and deploy a model to an Azure Kubernetes Service (AKS) cluster for real-time inferencing. You can now return to the Udacity portal to continue with the lesson
6.19 Walkthrough: Deploy a Model as a Webservice
https://photos.app.goo.gl/ZUx32rZxf3SJsVQJ8
https://photos.app.goo.gl/ynfrYEFX8xxNjnX67
https://photos.app.goo.gl/KLr4u6DkS69izD7M9
6.20 Prelaunch Lab
6.21 Programmatically Accessing Managed Services
Azure Machine Learning provides a code-first experience via the Azure Machine Learning SDK for Python. Using the SDK, you can start training your models on your local machine and then scale out to use Azure Machine Learning compute resources. This allows you to train better performing, highly accurate machine learning models.
https://photos.app.goo.gl/bYmvSWfp5mcjywXA8
Azure Machine Learning service supports many of the popular open-source machine learning and deep learning Python packages that we discussed earlier in the course, such as:
•	Scikit-learn
•	Tensorflow
•	PyTorch
•	Keras
In the next lab, we'll get some practice using Azure ML Python SDK to register, package, and deploy a trained model.
6.22 Lab: Training and Deploying from a Compute Instance
Compute Resources

Training and deploying a model from a notebook running in a Compute Instance
So far, the Managed Services for Azure Machine Learning lesson has covered compute instance and the benefits it provides through its fully managed environment containing everything you need to run Azure Machine Learning.
The compute instance provides a comprehensive set of a capabilities that you can use directly within a python notebook or python code including:
•	Creating a Workspace that acts as the root object to organize all artifacts and resources used by Azure Machine Learning.
•	Creating Experiments in your Workspace that capture versions of the trained model along with any desired model performance telemetry. Each time you train a model and evaluate its results, you can capture that run (model and telemetry) within an Experiment.
•	Creating Compute resources that can be used to scale out model training, so that while your notebook may be running in a lightweight container in Azure Notebooks, your model training can actually occur on a powerful cluster that can provide large amounts of memory, CPU or GPU.
•	Using Automated Machine Learning (AutoML) to automatically train multiple versions of a model using a mix of different ways to prepare the data and different algorithms and hyperparameters (algorithm settings) in search of the model that performs best according to a performance metric that you specify.
•	Packaging a Docker Image that contains everything your trained model needs for scoring (prediction) in order to run as a web service.
•	Deploying your Image to either Azure Kubernetes or Azure Container Instances, effectively hosting the Web Service.
Overview
In this lab, you start with a model that was trained using Automated Machine Learning. Learn how to use the Azure ML Python SDK to register, package, and deploy the trained model to Azure Container Instances (ACI) as a scoring web service. Finally, test the deployed model (1) by make direct calls on service object, (2) by calling the service end point (Scoring URI) over http.
Exercise 1: Run the Notebook for this Lab
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	From the studio, navigate to Compute. Next, for the available Compute Instance, under Application URI select Jupyter. Be sure to select Jupyter and not JupterLab.
 
5.	From within the Jupyter interface, select New, Terminal.
 
6.	In the new terminal window run the following command and wait for it to finish:
git clone https://github.com/solliancenet/udacity-intro-to-ml-labs.git
 
7.	From within the Jupyter interface, navigate to directory udacity-intro-to-ml-labs/aml-visual-interface/lab-22/notebook and open deployment-with-AML.ipynb. This is the Python notebook you will step through executing in this lab.
 
8.	In the Setup portion of the notebook, you will be asked to provide values for subscription_id, resource_group, workspace_name, and workspace_region. To find these, open your Azure Machine Learning workspace in the Azure portal and copy the values as shown:
 
9.	Follow the instructions within the notebook to complete the lab.
Next Steps
Congratulations! You have just learned how to use the Jupyter application on a compute instance to deploy a trained model to Azure Container Instances (ACI) for real-time

6.23 Walkthrough: Training and Deploying a Model from a Notebook Running in a Compute Instance
https://photos.app.goo.gl/eLsEWNrASf7x7QUS7
6.24 Lesson Summary
https://photos.app.goo.gl/xYrGpUV44B5mawGx9
In this lesson you've learned about managed services for Machine Learning and how these services are used to enhance Machine Learning processes.
First, you learned about various types of computing resources made available through managed services, including:
•	Training compute
•	Inferencing compute
•	Notebook environments
Next, you studied the main concepts involved in the modeling process, including:
•	Basic modeling
•	How parts of the modeling process interact when used together
•	More advanced aspects of the modeling process, like automation via pipelines and end-to-end integrated processes (also known as DevOps for Machine Learning or simply, MLOps)
•	How to move the results of your modeling work to production environments and make them operational
Finally, you were introduced to the world of programming the managed services via the Azure Machine Learning SDK for Python.

7 Responsible AI
7.1 Lesson Overview
https://photos.app.goo.gl/Q6pMGJDV9kuJ7hgg7
This lesson will introduce you to the essential and difficult discussion about the potential implications and challenges posed by Machine Learning. In this lesson, we will explore:

•	The modern-day challenges posed by AI in general and Machine Learning in particular.
•	The core principles of responsible AI (as a broader perspective of Machine Learning).
•	How Microsoft applies these principles.
•	Two essential aspects of Machine Learning models that impact responsible AI: transparency and explainability.

7.2 Modern AI: Challenges and Principles
https://photos.app.goo.gl/s5YiVT7wYKgNEFqF7

 
7.3 Microsoft AI Principles
https://photos.app.goo.gl/VYfvGtQQovpSRpR16

 
 

7.4 Prelaunch Lab
7.5 Model Transparency and Explainability
https://photos.app.goo.gl/6uxovxe4Hg8F9EV6A
https://photos.app.goo.gl/MpVe2Mo6BhKPPpuX8
How to Understand and Explain Models
 7.6 Lab: Model Explainability
Explaining models

Model interpretability with Azure Machine Learning service
Machine learning interpretability is important in two phases of machine learning development cycle:
•	During training: Model designers and evaluators require interpretability tools to explain the output of a model to stakeholders to build trust. They also need insights into the model so that they can debug the model and make decisions on whether the behavior matches their objectives. Finally, they need to ensure that the model is not biased.
•	During inferencing: Predictions need to be explainable to the people who use your model. For example, why did the model deny a mortgage loan, or predict that an investment portfolio carries a higher risk?
The Azure Machine Learning Interpretability Python SDK incorporates technologies developed by Microsoft and proven third-party libraries (for example, SHAP and LIME). The SDK creates a common API across the integrated libraries and integrates Azure Machine Learning services. Using this SDK, you can explain machine learning models globally on all data, or locally on a specific data point using the state-of-art technologies in an easy-to-use and scalable fashion.
Overview
In this lab, we will be using a subset of NYC Taxi & Limousine Commission - green taxi trip records available from Azure Open Datasets. The data is enriched with holiday and weather data. We will use data transformations and the GradientBoostingRegressor algorithm from the scikit-learn library to train a regression model to predict taxi fares in New York City based on input features such as, number of passengers, trip distance, datetime, holiday information and weather information.
The primary goal of this quickstart is to explain the predictions made by our trained model with the various Azure Model Interpretability packages of the Azure Machine Learning Python SDK.
Exercise 1: Run the Notebook for this Lab
1.	In Azure portal, open the available machine learning workspace.
2.	Select Launch now under the Try the new Azure Machine Learning studio message.
 
3.	When you first launch the studio, you may need to set the directory and subscription. If so, you will see this screen:
 
For the directory, select Udacity and for the subscription, select Azure Sponsorship. For the machine learning workspace, you may see multiple options listed. Select any of these (it doesn’t matter which) and then click Get started.
4.	From the studio, navigate to Compute. Next, for the available Compute Instance, under Application URI select Jupyter. Be sure to select Jupyter and not JupterLab.
 
5.	From within the Jupyter interface, select New, Terminal.
 
6.	In the new terminal window run the following command and wait for it to finish:
git clone https://github.com/solliancenet/udacity-intro-to-ml-labs.git
 
7.	From within the Jupyter interface, navigate to directory udacity-intro-to-ml-labs/aml-visual-interface/lab-23/notebook and open interpretability-with-AML.ipynb. This is the Python notebook you will step through executing in this lab.
 
8.	Follow the instructions within the notebook to complete the lab.
Next Steps
Congratulations! You have just learned how to use the Azure Machine Learning SDK to help you explain what influences the predictions a model makes. You can now return to the Udacity portal to continue with the lesson.
7.7 Walkthrough: Model Explainability
https://photos.app.goo.gl/YDdEXfZskiPJ3WNw8
https://photos.app.goo.gl/piMwGmtq1hrDaJ1p6
7.8 Model Fairness
https://photos.app.goo.gl/6SKnQLCihh2F1MTh7
In this next video, we'll walk through an example of how you could use some of the Fairlearn capabilities when applying model fairness to your work. This demo is just intended as an example, not a lab exercise, although you are certainly welcome to check out the FairLearn repository for yourself, including the notebooks we use in the demo, here.
Fairlearn Notebook Demo
https://photos.app.goo.gl/NzBKWrn2AN3bduo2A

7.9 Lesson Summary
https://photos.app.goo.gl/R8L7Z9fiR4FCkX5s8
In this lesson you've learned about the potential implications and difficult challenges posed by Machine Learning.
You've studied the core principles of responsible AI and how Microsoft aligns its AI strategy according to them. You've also learned about model transparency and explainability, and you've performed an exploration of model explanations using Azure Machine Learning Studio.
Finally, you've learned about some of the modern-day challenges posed by AI and Machine Learning.
8.1 Course Conclusion
https://photos.app.goo.gl/dmU8wuQNrG4B9nEN7
The Future and Importance of Machine Learning
https://photos.app.goo.gl/CnjadamUC1YF1TqPA
•	Machine Learning is already having a significant impact on almost every aspect of our daily lives
•	Machine Learning promises to help advance critically important fields like medicine, transportation, space travel, agriculture, cybersecurity, and many more
•	The trends in computing power availability, moving closer to silicon, public cloud HW resource concentration will continue to accelerate in the next decade
•	Other medium or long-term scientific breakthroughs (think quantum computing) have the potential of creating even larger ML advancements
•	Better understanding of current algorithms, inventing new ones, focusing on transparency and explainability will be also major trends



Course recap
This course was meant to be an introduction to the fantastic world of Machine Learning. Here is a short recap of our journey:
In Lesson 1, you've learned the fundamentals of Machine Learning, made your first contact with data, models, and algorithms, and trained your first model in Azure Machine Learning Studio.
In Lesson 2, you performed a few basic data preparation tasks like import, transformation, and feature engineering. You've also trained a classifier and a regressor and learned to apply the "strength in numbers" principle using ensembles and automated Machine Learning.
In Lesson 3, you've learned about supervised and unsupervised learning, two of Machine Learning's fundamental approaches. You've also trained and evaluated the performance of classifiers, regressors, and clustering models.
In Lesson 4, you got your first taste of Deep Learning and dived into some of the most important specialized cases like similarity learning, text classification, feature learning, anomaly detection, and forecasting.
In Lesson 5, you've learned to manage compute resources, model training and evaluation processed, and operational environment. You've also experienced the use of the Azure Machine Learning SDK for Python to program the Azure Machine Learning managed services.
Finally, in Lesson 6, you've learned the core principles of responsible AI. You've also learned to train transparent and explainable models, and raised your awareness on some of the serious challenges of modern AI and Machine Learning.



