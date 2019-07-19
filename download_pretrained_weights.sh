echo "downloading pretrained model..."

MODEL=./pre_trained_weights.zip
URL_MODEL=https://collections.durham.ac.uk/downloads/r25425k9734

echo "downloading the model weights..."

wget --quiet --no-check-certificate --show-progress $URL_MODEL -O $MODEL

echo "checking the MD5 checksum for downloaded model..."

CHECK_SUM_CHECKPOINTS='e955db2cf2bc11a3e79e2d2153299fce  pre_trained_weights.zip'

echo $CHECK_SUM_CHECKPOINTS | md5sum -c

echo "Unpacking the zip file..."

unzip -q pre_trained_weights.zip && rm pre_trained_weights.zip 

cd pre_trained_weights

rm README.txt

echo "All Done!!"
