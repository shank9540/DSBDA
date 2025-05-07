package com.wc;

import java.io.IOException;
import java.util.StringTokenizer;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;

public class WC_Mapper extends MapReduceBase implements Mapper<LongWritable, Text, Text, IntWritable> {
    private final static IntWritable one = new IntWritable(1);  // Constant value for counting
    private Text word = new Text();  // Text object to hold each word

    public void map(LongWritable key, Text value, OutputCollector<Text, IntWritable> output, Reporter reporter) throws IOException {
        // Convert each line to a string
        String line = value.toString();
        StringTokenizer tokenizer = new StringTokenizer(line);

        // Tokenize and collect each word
        while (tokenizer.hasMoreTokens()) {
            word.set(tokenizer.nextToken());  // Set word to the next token
            output.collect(word, one);  // Emit word and count 1
        }
    }
}
