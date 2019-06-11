# 1. Download Data

First, login to your Linux machine. 
Follow instructions for [JSALT19 summer school](https://gist.github.com/jtrmal/40504cf2731f47bea6b6f75c468aa999)

Next, we download a pre-processed learning-to-rank training/dev/test data. The original source of this data is OHSUMED in [Microsoft's LETOR3.0 dataset](http://research.microsoft.com/en-us/um/beijing/projects/letor/letor3download.aspx). We've taken a subset of it for fast training.


```
~$ mkdir ltr; cd ltr
~/ltr$ wget http://www.cs.jhu.edu/~kevinduh/t/letor.tgz
~/ltr$ tar -xvf letor.tgz
```


# 2. Understand the Data Format

Format: relevance label, query id, features+

```
~/ltr$ head -2 letor/train.txt
0 qid:1 1:1.000000 2:1.000000 3:0.833333 4:0.871264 5:0 6:0 7:0 8:0.941842 9:1.000000 10:1.000000 11:1.000000 12:1.000000 13:1.000000 14:1.000000 15:1.000000 16:1.000000 17:1.000000 18:0.719697 19:0.729351 20:0 21:0 22:0 23:0.811565 24:1.000000 25:0.972730 26:1.000000 27:1.000000 28:0.922374 29:0.946654 30:0.938888 31:1.000000 32:1.000000 33:0.711276 34:0.722202 35:0 36:0 37:0 38:0.798002 39:1.000000 40:1.000000 41:1.000000 42:1.000000 43:0.959134 44:0.963919 45:0.971425 #docid = 244338
2 qid:1 1:0.600000 2:0.600000 3:1.000000 4:1.000000 5:0 6:0 7:0 8:1.000000 9:0.624834 10:0.767301 11:0.816099 12:0.934805 13:0.649685 14:0.680222 15:0.686762 16:0.421053 17:0.680904 18:1.000000 19:1.000000 20:0 21:0 22:0 23:1.000000 24:0.401391 25:0.938966 26:0.949446 27:0.984769 28:0.955266 29:1.000000 30:0.997786 31:0.441860 32:0.687033 33:1.000000 34:1.000000 35:0 36:0 37:0 38:1.000000 39:0.425450 40:0.975968 41:0.928785 42:0.978524 43:0.979553 44:1.000000 45:1.000000 #docid = 143821
```


Questions to check your understanding: 
* How many documents in train.txt?
* How many queries?
* How many documents/query?

# 3. Try out an existing Ranker

We will first install [SVM-Rank](SVMrank: https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html). 

```
~/ltr$ mkdir svm_rank
kduh@master:~/ltr$ cd svm_rank/
kduh@master:~/ltr/svm_rank$ wget http://download.joachims.org/svm_rank/current/svm_rank.tar.gz
~/ltr/svm_rank$ tar -xvf svm_rank.tar.gz
~/ltr/svm_rank$ make
```

This will produce the executables `svm_rank_learn` for training and `svm_rank_classify` for test-time inference. Read the HELP manual:

```
~/ltr/svm_rank$ ./svm_rank_learn -?

SVM-struct learning module: SVM-rank, V1.00, 15.03.2009
   includes SVM-struct V3.10 for learning complex outputs, 14.08.08
   includes SVM-light V6.20 quadratic optimizer, 14.08.08

Copyright: Thorsten Joachims, thorsten@joachims.org

This software is available for non-commercial use only. It must not
be modified and distributed without prior permission of the author.
The author is not responsible for implications from the use of this
software.

   usage: svm_struct_learn [options] example_file model_file

Arguments:
         example_file-> file with training data
         model_file  -> file to store learned decision rule in
General Options:
         -?          -> this help
         -v [0..3]   -> verbosity level (default 1)
         -y [0..3]   -> verbosity level for svm_light (default 0)
Learning Options:
         -c float    -> C: trade-off between training error
                        and margin (default 0.01)
         -p [1,2]    -> L-norm to use for slack variables. Use 1 for L1-norm,
                        use 2 for squared slacks. (default 1)
         -o [1,2]    -> Rescaling method to use for loss.
                        1: slack rescaling
                        2: margin rescaling
                        (default 2)
         -l [0..]    -> Loss function to use.
                        0: zero/one loss
                        ?: see below in application specific options
                        (default 1)
Optimization Options (see [2][5]):
         -w [0,..,9] -> choice of structural learning algorithm (default 3):
                        0: n-slack algorithm described in [2]
                        1: n-slack algorithm with shrinking heuristic
                        2: 1-slack algorithm (primal) described in [5]
                        3: 1-slack algorithm (dual) described in [5]
                        4: 1-slack algorithm (dual) with constraint cache [5]
                        9: custom algorithm in svm_struct_learn_custom.c
         -e float    -> epsilon: allow that tolerance for termination
                        criterion (default 0.001000)
         -k [1..]    -> number of new constraints to accumulate before
                        recomputing the QP solution (default 100)
                        (-w 0 and 1 only)
         -f [5..]    -> number of constraints to cache for each example
                        (default 5) (used with -w 4)
         -b [1..100] -> percentage of training set for which to refresh cache
                        when no epsilon violated constraint can be constructed
                        from current cache (default 100%) (used with -w 4)
SVM-light Options for Solving QP Subproblems (see [3]):
         -n [2..q]   -> number of new variables entering the working set
                        in each svm-light iteration (default n = q).
                        Set n < q to prevent zig-zagging.
         -m [5..]    -> size of svm-light cache for kernel evaluations in MB
                        (default 40) (used only for -w 1 with kernels)
         -h [5..]    -> number of svm-light iterations a variable needs to be
                        optimal before considered for shrinking (default 100)
         -# int      -> terminate svm-light QP subproblem optimization, if no
                        progress after this number of iterations.
                        (default 100000)
Kernel Options:
         -t int      -> type of kernel function:
                        0: linear (default)
                        1: polynomial (s a*b+c)^d
                        2: radial basis function exp(-gamma ||a-b||^2)
                        3: sigmoid tanh(s a*b + c)
                        4: user defined kernel from kernel.h
         -d int      -> parameter d in polynomial kernel
         -g float    -> parameter gamma in rbf kernel
         -s float    -> parameter s in sigmoid/poly kernel
         -r float    -> parameter c in sigmoid/poly kernel
         -u string   -> parameter of user defined kernel
Output Options:
         -a string   -> write all alphas to this file after learning
                        (in the same order as in the training set)
Application-Specific Options:

The following loss functions can be selected with the -l option:
     1  Total number of swapped pairs summed over all queries.
     2  Fraction of swapped pairs averaged over all queries.

NOTE: SVM-light in '-z p' mode and SVM-rank with loss 1 are equivalent for
      c_light = c_rank/n, where n is the number of training rankings (i.e.
      queries).

The algorithms implemented in SVM-perf are described in:
- T. Joachims, A Support Vector Method for Multivariate Performance Measures,
  Proceedings of the International Conference on Machine Learning (ICML), 2005.
- T. Joachims, Training Linear SVMs in Linear Time, Proceedings of the
  ACM Conference on Knowledge Discovery and Data Mining (KDD), 2006.
  -> Papers are available at http://www.joachims.org/
```

For training, the basic usage is:
`svm_struct_learn [options] example_file model_file`
where `example_file` is the input training data and `model_file` is the path of the output model after training.

For test-time inference, the basic usage is: 
`svm_struct_classify [options] example_file model_file output_file`,
where `example_file` is the input test data, `model_file` is the path of the trained model, and `output_file` is the path of the output inference.

## Training SVM-Rank: 

Let's try one run:

```
~/ltr/svm_rank$ cd ..
~/ltr$ ./svm_rank/svm_rank_learn -c 20 letor/train.txt model1
...
Total number of constraints in final working set: 88 (of 223)
Number of iterations: 224
Number of calls to 'find_most_violated_constraint': 14112
Number of SV: 24
Norm of weight vector: |w|=15.44728
Value of slack variable (on working set): xi=4696.59435
Value of slack variable (global): xi=4699.83906
Norm of longest difference vector: ||Psi(x,y)-Psi(x,ybar)||=302.04911
Runtime in cpu-seconds: 1.94
Compacting linear model...done
Writing learned model...done
```

Let's look inside the trained model. In this case, it's a linear model, so it's easy to interpret the weights. 

```
~/ltr$ cat model1
SVM-light Version V6.20
0 # kernel type
3 # kernel parameter -d
1 # kernel parameter -g
1 # kernel parameter -s
1 # kernel parameter -r
empty# kernel parameter -u
46 # highest feature index
88 # number of training documents
2 # number of support vectors plus 1
0 # threshold b, each following line is a SV (starting with alpha*y)
1 1:1.9970672 2:-3.0596602 3:0.064775988 4:0.85808605 8:-0.21327031 9:0.0098601459 10:0.10698769 11:1.6736647 12:0.29681849 13:0.10356258 14:-1.6108279 15:1.6323735 16:0.44374746 17:-3.9472368 18:-2.3306632 19:0.57510704 23:1.2300572 24:4.1410909 25:-0.54958433 26:1.3788967 27:-0.95501417 28:1.6788845 29:-1.4944086 30:0.83200067 31:2.2233305 32:3.0258162 33:6.9870725 34:-7.8474011 38:0.97317785 39:-6.0868015 40:-0.79811758 41:1.1158154 42:-0.097357437 43:1.3300766 44:-0.92236727 45:-0.49284962 #
```

Question: what features do you think contribute most to high rankings? 


## Testing SVM-Rank

Let's evaluate on the validation data:
```
~/ltr$ ./svm_rank/svm_rank_classify letor/vali.txt model1 vali.pred
Reading model...done.
Reading test examples...done.
Classifying test examples...done
Runtime (without IO) in cpu-seconds: 0.00
Average loss on test set: 0.3840
Zero/one-error on test set: 100.00% (0 correct, 21 incorrect, 21 total)
NOTE: The loss reported above is the fraction of swapped pairs averaged over
      all rankings. The zero/one-error is fraction of perfectly correct
      rankings!
Total Num Swappedpairs  :  51224
Avg Swappedpairs Percent:  38.40
```

The output is saved in `vali.pred`. Note these are scores for each query-document pair, concatenated over multiple queries. For example, the first query (qid:64) is in lines 1 to 188, while the second query (qid:65) starts in line 189.  

```
~/ltr$ cat -n vali.pred
     1	0.79657811
     2	2.28742832
     3	2.29528763
     4	0.98297993
     5	1.02327700
     6	1.12129519
     7	1.48147723
     8	1.26908941
...
~/ltr$ awk '{print $2}' letor/vali.txt  |cat -n
     1  qid:64
     2  qid:64
     3  qid:64
     4  qid:64
     5  qid:64
     6  qid:64
     7  qid:64
     ...
     187  qid:64
     188  qid:64
     189  qid:65
     190  qid:65
```

Finally, we can compute standard information retrieval metrics, such as Mean Averaged Precision (MAP):

```
~/ltr$ perl letor/Eval-Score-3.0.pl letor/vali.txt vali.pred vali.pred.result 0
```

The 0 specifies we only want overall metrics; specifying 1 will give you query-level metrics. Here, `letor/vali.txt` will be consulted as the gold reference, and the MAP/NDCG/etc of `vali.pred` is  output in `vali.pred.result`:

```
cat vali.pred.result
precision:	0.666666666666667	0.69047619047619	0.682539682539682	0.654761904761905	0.628571428571429	0.603174603174603	0.578231292517007	0.56547619047619	0.544973544973545	0.538095238095238	0.515151515151515	0.503968253968254	0.512820512820513	0.513605442176871	0.514285714285714	0.508928571428571

MAP:	0.458439033060933

NDCG:	0.571428571428571	0.531746031746032	0.518423116936691	0.501539201844428	0.497504441175727	0.485304720847943	0.475108708550683	0.472233559725649	0.464736229217328	0.460632952964679	0.452592822879271	0.450291682694412	0.456612453602719	0.453432209009918	0.454673283432685	0.453452539422857
```

MAP is	0.45. Congratulations on building and testing your first learning-to-rank model!

# 4. Explore more

Here are several options for further exploration:

1. Implement your own: 
  * point-wise method with linear regression, using sklearn.linear_model.LinearRegression
  * Implement pair-wise method with binary classification, using sklearn.linear_model.LogisticRegression

2. Learn more about existing toolkits:
  * Read more about [SVMrank](https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html)
  * Try the various methods in [RankLib](https://sourceforge.net/p/lemur/wiki/RankLib/) 
  * TensorFlow [TF-Ranking](https://ai.googleblog.com/2018/12/tf-ranking-scalable-tensorflow-library.html)
  * Many more!

3. Evaluate and compare MAP results


 
