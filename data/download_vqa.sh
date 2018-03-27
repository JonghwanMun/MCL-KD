#!/bin/sh

# downloaded files will be moved to vqa_v1.9 folder.
echo "Download vqa v2.0 datasest"
if [ ! -d "vqa_v2.0/annotations" ]; then
	mkdir -p vqa_v2.0/annotations
	mkdir -p vqa_v2.0/images
fi
# vqa training/validation annotations 2017 v2.0
wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip v2_Annotations_Train_mscoco.zip
rm -rf v2_Annotations_Train_mscoco.zip
unzip v2_Annotations_Val_mscoco.zip
rm -rf v2_Annotations_Val_mscoco.zip
mv v2_mscoco_train2014_annotations.json vqa_v2.0/annotations/
mv v2_mscoco_val2014_annotations.json vqa_v2.0/annotations/

# vqa training/validation/test questions 2017 v2.0
wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/v2_Questions_Test_mscoco.zip
unzip v2_Questions_Train_mscoco.zip
rm -rf v2_Questions_Train_mscoco.zip
unzip v2_Questions_Val_mscoco.zip
rm -rf v2_Questions_Val_mscoco.zip
unzip v2_Questions_Test_mscoco.zip
rm -rf v2_Questions_Test_mscoco.zip
mv v2_OpenEnded_mscoco_train2014_questions.json vqa_v2.0/annotations/
mv v2_OpenEnded_mscoco_val2014_questions.json vqa_v2.0/annotations/
mv v2_OpenEnded_mscoco_test2015_questions.json vqa_v2.0/annotations/
mv v2_OpenEnded_mscoco_test-dev2015_questions.json vqa_v2.0/annotations/

# vqa images
# vqa training/validation/test images 2017 v2.0
#   wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
#   wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
#   wget http://msvocds.blob.core.windows.net/coco2015/test2015.zip
#   unzip train2014.zip
#   rm -rf train2014.zip
#   unzip val2014.zip
#   rm -rf val2014.zip
#   unzip test2014.zip
#   rm -rf test2014.zip
#   mv train2014 vqa_v2.0/images
#   mv val2014 vqa_v2.0/images
#   mv test2015 vqa_v2.0/images

# vqa training/validation/test complementary 2017 v2.0
wget http://visualqa.org/data/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip
unzip v2_Complementary_Pairs_Train_mscoco.zip
rm -rf v2_Complementary_Pairs_Train_mscoco.zip
unzip v2_Complementary_Pairs_Val_mscoco.zip
rm -rf v2_Complementary_Pairs_Val_mscoco.zip
mv v2_mscoco_train2014_complementary_pairs.json vqa_v2.0/annotations/
mv v2_mscoco_val2014_complementary_pairs.json vqa_v2.0/annotations/
