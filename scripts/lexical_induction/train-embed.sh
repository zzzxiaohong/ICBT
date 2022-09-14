work_dir= # your working directory path     
tools_dir=$work_dir/tools
fasttext=$tools_dir/fastText/build/fasttext

# # combine unaligned monolingual data in both languages
mono_data_dir=$work_dir/data/datamono
dout_data_dir=$work_dir/data/dout_data

dest_dir=$work_dir/output/embed
mkdir -p $dest_dir
cat $data_dir/*-train.tok.mono.en > $data_dir/all-train.mono.en
cat $data_dir/*-train.tok.mono.zh > $data_dir/all-train.mono.zh
cat $sdata_dir/train.tok.en >> $data_dir/all-train.mono.en
cat $sdata_dir/train.tok.zh >> $data_dir/all-train.mono.zh


for lang in en zh
    do  
        if [ ! -f ${dest_dir}/all-train.mono.${lang} ]
        then
            echo "$fasttext skipgram -input ${data_dir}/all-train.mono.${lang} -output ${dest_dir}/all-train.mono.${lang} -ws 10 -dim 512 -neg 5 -t 1e-4 -epoch 10"
            $fasttext skipgram -input ${data_dir}/all-train.mono.${lang} -output ${dest_dir}/all-train.mono.${lang} \
                -dim 512 -lr 0.025 -ws 5 -epoch 10 -minCount 5 -neg 5 -loss ns -bucket 2000000 \
                -minn 3 -maxn 6 -thread 4 -t 1e-4 -lrUpdateRate 100
        else
            echo "file has already exisit!!"
        fi
done
