# Exploring the coherence of features explanations in the GemmaScope

Mattia Proietti [AI Alignment course project (BlueDot Impact)](https://bluedot.org/courses/alignment) autumn 2024

Code for my project for the course on AI safety by BlueDot impact.
A full write-up of the project can be found as a LessWrong post at [Exploring the coherence of features explanations in the GemmaScope](https://www.lesswrong.com/posts/YEJinznLzNtLfE67m/exploring-the-coherence-of-features-explanations-in-the)

# Summary
Sparse Autoencoders are becoming the to-use technique to get features out of superposition and interpret them in a sparse activation space. Current methods to interpret such features at scale rely heavily on automatic evaluations carried by Large Language Models. However, there is still a lack of understanding of the degree to which these explanations are coherent both with the input text responsible for their activation and the conceptual structure for which such inputs stand for and represent.

We devise a very naive and incipient heuristic method to explore the coherence of explanations provided in the GemmaScope with a set of controlled input sentences regarding animals, leveraging the simple conceptual structure derivable in terms of simple taxonomic relations triggered by specific animal words (e.g.. The dog is a mammal).

We show how, for the input we used and the location we analysed, the explanations of features at 5 different strength levels of activation are largely irrelevant and incoherent with respect to the target words and the conceptual structure they leverage


