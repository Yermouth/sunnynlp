This directory contains data that can be used for model training and parameter tuning in SemEval 2018 Task 10: Capturing Discriminative Attributes.

Data format: 4 comma-separated fields:
- word 1 (pivot)
- word 2 (comparison)
- feature
- label ("1" if the feature characterizes word 1 compared to word 2, "0" otherwise)

train.txt contains 17782 examples with a total of 1292 distinct features, including:
- 11191 negative examples with a total of 1290 distinct features
- 6591 positive examples with a total of 333 distinct features

validation.txt contains 2722 examples with a total of 576 distinct features, including:
- 1364 positive examples with a total of 410 distinct features
- 1358 negative examples with a total of 383 distinct features

Sample of data in train.txt:

basket,parsley,picked,0
oven,microwave,control,1
moose,elk,waxing,0
harpoon,missile,inexpensive,0
corn,pineapple,ears,1
panther,elephant,pillows,0
belt,plate,buckles,1
orange,cherry,sections,1
razor,brush,mink,0
necklace,bracelet,clasp,0


Sample of data in validation.txt:

bat,cat,wings,1
sofa,pillow,legs,1
arm,foot,bones,0
tangerine,avocado,round,1
fingers,cheek,bones,1
meat,potatoes,protein,1
cigar,biscuit,long,0
condos,motel,apartment,1
educator,physician,educated,0
clouds,snow,weather,0
