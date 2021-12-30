mkdir $1
cp $1.py ./$1/lambda_function.py
pip install --target ./$1 ${@:2}
cd ./$1
zip -r ../$1.zip .
cd ..
rm -rf ./$1

