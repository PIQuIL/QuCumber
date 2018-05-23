#!/bin/sh

#conda activate py3all

output_file='pytorch_cpu.csv'
#rm ${output_file}

echo "L,epochs,batch_size,k,time,memory" >> ${output_file}

for size in 32; do
for epoch in 250; do
for batch in 32; do
for k in 1 10; do
    
    echo "Iterating..."

    echo -n "$size," >> ${output_file}
    echo -n "$epoch," >> ${output_file}
    echo -n "$batch," >> ${output_file}
    echo -n "$k," >> ${output_file}

    /usr/bin/time -f "%e,%M" -o ${output_file} -a  \
	python pytorch/run_RBM.py $size $batch $epoch $size $k 0.001 0.9 0 \
	> /dev/null 2>&1

done
done
done
done

output_file='pytorch_gpu.csv'
#rm ${output_file}

echo "L,epochs,batch_size,k,time,memory" >> ${output_file}

for size in 32; do
for epoch in 250; do
for batch in 32; do
for k in 1 10; do
    
    echo "Iterating..."

    echo -n "$size," >> ${output_file}
    echo -n "$epoch," >> ${output_file}
    echo -n "$batch," >> ${output_file}
    echo -n "$k," >> ${output_file}

    /usr/bin/time -f "%e,%M" -o ${output_file} -a  \
	python pytorch/run_RBM.py $size $batch $epoch $size $k 0.001 0.9 1 \
	> /dev/null 2>&1

done
done
done
done
