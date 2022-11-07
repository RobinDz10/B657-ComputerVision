## Description
This assignment involves creating modules for a cheating-proof final exam that could be graded perfectly and effortlessly.

## How to run code

### Use the following command for running the grade.py:
python3 grade.py form.jpg output.txt

### Use the following command for running the inject.py:
python3 inject.py form.jpg answers.txt injected.jpg

### Use the following command for running the extract.py:
python3 extract.py injected.jpg output.txt

## Assumptions made
Following are the assumptions made:
1. All the questions will have at least one answer option in the answers.txt file for a blank form.
2. There would be no extreme rotation or resizing of the image.

## Problem 1
### Approach (Kernel Based approach)
Our overall approach for problem 1 is as follows:
- Find the beginning of each question by running a square kernel all over the image.
- Once we determine, that a question is present at a location, we check which options are selected, by running another 
  kernel on all 5 option boxes and evaluating whether it crosses a certain threshold.
  
Other smaller details of what decisions did we take are listed here.

**Find all the option (square) boxes.**

To find all option boxes, we created a kernel of 35 x 33, which will have highest activations, when it encounters a box with black edges.
We ran this kernel all over the image to identify the points of interest.

**Take max 1000 activations out of all the activations at every pixel of input image**

Since there are 85 questions, so there will be at max 85 x 5 + 10(sample boxes on top) = 435 boxes that we need to detect.
But to account misc. variations in the image, we extracted top 1000, and we later attempt to remove some of these points to reduce this number to 435.

**Remove very close points in these 1000 points (suppression), and keep only 435 points**

While removing/keeping the points, we follow from highest activations to lowest activation, in these 1000 points. That is, 
if two activations are close by, we keep the higher activation for consideration and remove the other point. To define proximity, 
we kept a configurable parameter window and if two points lie in same window, then one of them is removed. 

**For each point in the remaining, group points, based on whether that point belongs to first option box of the question.**

To check if the point is the first option box of the question, we check if there are few other points after that point within a
range of (<avg option box distance> +- <tolerance>). If we find a total of three points, then we keep consider that point to be 
first option box. We limit it to 3, because some of the option boxes are not detected in the 1000 points, so we allow some slack.
Another thing we do is, we start from left-top to bottom-right order, to see if the option box is the first option box.

**Combine the question beginnings(first option box) together if they are close by and finally form a cluster**

We combine two question beginnings, if they are within a certain distance(avg inter option box distance vertical), with some tolerance.
While combining, we add missing option boxes that might have not been detected. The process looks like below:

o   o   o   o   o

o   o   o   ~   ~   

*Changes to*

o   o   o   o   o

o   o   o   o   o

**Combine very close groups**

o   o   o   o   o

o   o   o   o   o

~   ~   ~   ~   ~

o   o   o   o   o

o   o   o   o   o

*Notice that missing rows were added when combining two groups*
*In some cases the groups are adjusted/shifted*

**Choose the groups of size more than 1(Bottom three)**

**For each question in each vertical group Check which options are selected**

For this, we run a white kernel over the each of the five option boxes at fixed distances apart and see if it is above a threshold
to determine if the option box is selected.

**Based on which options are selected, we generate the output file of selected answers by the student.**


## Problem 2

We attempted 3 approaches

### Approach 1 - FFT based approach
Observing how modifications made in FFT domain are almost impossible to decipher, this approach was of interest to us.
Our overall approach was to somehow make changes in the FFT of the image, and then while extracting the ground truth, 
we planned to retrieve those changes. Another motivation for this was the secret message ('HI') module, which showed as
to how can a message be hidden in the image without a person knowing about it. This served as the basis for hiding the
answers of the questions.

For the Injection and Extraction of the answers, the Fourier Transform approach was tried in which the image in the 
spatial domain is transformed to the image in the frequency domain.

Since there were 85 questions and 5 possible answers per question, the idea was to uniquely determine 425 points in the 
Fourier space such that each of those points correspond to the possible answers to the questions of the blank form.

A block of 875x625 was identified in the Fourier space that was away from the center and corners. This block was divided
in horizontal bands of five classes (A, B, C, D, and E). Each band had 109375 points out of which 85 points were 
selected. The magnitude at those 85 points would represent if the question had that answer or not.

Following are the two phases:
#### 1. Injection Phase
Assume that Question 6 had ‘A' as an answer, then in the first band, a point (out of 85 selected 
points) that corresponded to the 6th Question would have been set to a low value. Since this was done in first band, 
this would signify that ‘A' was the answer to the 6th Question. Similarly, assume that the Question 10 had both ‘A’ and
‘C’ as the answer, then the points corresponding to Question 10 in first and third band would have been set to a low 
value.

#### 2. Extraction Phase
After the injected image is ready from the previous step, the same is scanned and supplied in the extraction phase. 
Here, we go from band number 1 to band number 5 locating all 85 points in each band and see as to where all the values 
are low. As and when we find low values, we create a mapping and finally, lay down all the mappings in an output file.

### Implementation Completion:
Injection Phase(with approach - 1): 100% implemented

Extraction Phase(with approach - 1): 90% implemented

There was a problem in the extraction phase. The low value at the expected points was not low at all. Even after setting
the low value in the injection phase, the value of those points in the extraction phase was almost of the same magnitude
as the points around it. This was a major roadblock that prevented the 100% implementation of the extraction phase, 
leaving it to 90% only.

To remove the roadblock, ideas from web (specifically StackOverflow) and the Q&A community  were tried along with 
various techniques such as setting higher values, dividing the values, scaling, using negative values, average values, 
setting value for the selected point as well as points around it, real points, png format, etc.

### Code
For both the phases, the code is available as part of the Branch Name: _fft-approach-for-inject-extract_ bearing 
Commit ID: _a2b9907_

### Approach 2 - Pixels modification approach
Create a list with its size 85 * 5 in the bottom-left corner of the answer sheet, due to there are 85 questions in total, and each question 
contains 5 answers. First, wipe out the whole bottome area of the image (the grayscale of the pixels in this area are all set to be 255), and 
then create the pixels list. Hard code each answer into the list, for example, if the answer of question 25 is ACD, then pixels[24][0] and 
pixels[24][2] and pixelsList[24][3] will be set to 0. After hardcoding correct answers, make a "mark" at the top of the pixels list, this will 
help us to read answers no matter the image being scanned a little bit left or right or up or downward. As for the extraction module, find the 
"mark" first and then read all the answers below. However, this solution will only help without the image being rotated in an angle.

### Approach 3 - Adding dots around the boundary

In this approach, we add dots around the rectangular boundary and during extraction, we read the points from boundary.
To account the effect of rotation, shifting and scaling, we add 4 fixed corner points. We use these points to determine 
the rotation/shift/scaling.

During the extraction procedure out main focus is to extract these 4 corner points. Once these 4 corner points are detected, 
all other points are mapped. 

Overall, there are two horizontal and two vertical lines. We add a total of 425(85 * 5) + 4(corner) = 429 points on these lines.
Which are all equi-distant on the lines.

**Readability of injected file**

The injected file that has been generated is not readable by a student. To further complicate it,
we jumbpled the order, so for example we write ground truths starting from question 1, at the bottom horizontal line and 
then right-most vertical line of the rectangular boundary.

We selected this approach finally, as it was invariant to rotation, scale and shift to some degree.

## Quantitative and Qualitative accuracy of the program on the test images
The program successfully runs on all the given test images with 100% accuracy.

## Highlights
Following are the highlights:
1. The approach is invariant to rotation if the angle of rotation ranges from -3.69 degree to +3.69 degree.
2. The approach is invariant to resizing if either the height and width of the scanned image is reduced from 1 to 0.64 
times or increased from 1 to 1.19 times.

## Improvements in future
Following are the improvements in future:
1. Putting patterns on the form takes away originality from it. Hence, we need to understand the functioning of Fourier 
Space better so that the answers can be hidden without any visible patterns.

## References
Following are the references:
1. https://homepages.inf.ed.ac.uk/rbf/HIPR2/fourier.htm
2. https://stackoverflow.com/questions/71180216/magnitude-of-a-few-points-in-fft2-for-a-2d-image-not-getting-retained-after-modi?noredirect=1#comment125824638_71180216
3. https://iu.instructure.com/courses/2032639/external_tools/271583
4. https://docs.scipy.org/doc/scipy/tutorial/fft.html
5. https://www.youtube.com/watch?v=GKsCWivmlHg&t=1645s

## Contributions of the Authors

### Duozhao Wang