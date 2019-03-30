PROBLEM:

What is the main problem of Indian Media? And who are the ones who get actually affected by the spreading of news, be it fake or not? It’s us, the common people. Whatever news we learn, we spread it to the nearest neighbours and so on.

SOLUTION:

We have come up with a solution to this, We haven’t named it as of yet (Hopefully by the end of this hackathon we will). Its basically brings news(any media) from all over the world and checks how trustworthy it is. 

Given India’s low literacy rate, images play a vital role in the spread of information. Images also have the ability to capture people’s attention quicker than text. This characteristic of a picture, unfortunately, can also be used to disseminate misinformation on a wide scale. The danger of fake information propagated through images is especially evident in India, especially in these pre-Election days wherein it has resulted in unwanted consequences, including death.

STEPS TO FIND THE FAKE MEDIA:

Viral social media posts and message forwards usually consist of an image and an associated title. Although the image might depict one a particular phenomenon or event, the text could purposefully be chosen to incite the viewer. For example, an image of two people having a physical altercation in which one person is wearing a skull cap can easily be misunderstood if the associated text were “Riots broken out at Faridabad - Muslims versus Hindus once again”. Although this was a hypothetical example, it is not impossible for us to envision a similar method of fake news propagation that could end up having fatal consequences.

We believe that curbing this subset of fake news would involve providing the necessary tools to journalists and users who take the time to sift through trustworthy content. In this light, our solution is to devise a methodology and program to assist a normal person to identify if an image and associated title are trustworthy. The fundamentals of our approach are based on corroboration of news, i.e, the image must be used in credible news websites with text similar to that of the associated title for the image and text to be reliable. If not, we recommend that the journalist (or user) perform a human verification procedure.

In our method, we first take an user image, which is then run through Google’s Vision API that performs a reverse image search that outputs page URLs of websites that have used the same picture as the user image.
The program searches through the list of page URLs to check if any of them are credible news sources(determined beforehand. The user or journalist has the ability to encode whichever source she believes is trustworthy within the program code as a list. We also provide a default credible list of news websites).

The program scrapes each URL from the list in order to extract the titles of each news article and is made to decipher the semantic similarity between the titles we have extracted and the original associated title. To perform this text meaning comparison, the program uses Google’s pre-trained news article text corpus which is trained on the Word2Vec algorithm.

In the next Steps, we are creating a platform to give the user and experience of trustworthy news and reduces the sharing of fake news or fake photos.

In order to be as helpful as possible to a user, our program provides the user a plethora of information to aid in manual verification.






