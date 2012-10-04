#!/bin/bash

TSVPATH=$1
SCRIPTPATH=$2
LOGPATH=$3
TSVFILE=$4
LOGFILE=$5

ONEFILE=$(ls $TSVPATH/run_*.tsv | head -1)
NUMFIELDS=$(head -1 $ONEFILE | sed "s/\t/\n/g" | wc -l)
if [ $NUMFIELDS -eq 14 ]; then
    echo -e "testset\tlexicon\tfeatureset\tlexicon_info\tclassifier\taccuracy\tS_accuracy\tO_accuracy\tS_precision\tS_recall\tS_f1\tO_precision\tO_recall\tO_f1" > $TSVFILE
elif [ $NUMFIELDS -eq 18 ]; then
    echo -e "testset\tlexicon\tfeatureset\tlexicon_info\tclassifier\taccuracy\tP_accuracy\tN_accuracy\tE_accuracy\tP_precision\tP_recall\tP_f1\tN_precision\tN_recall\tN_f1\tE_precision\tE_recall\tE_f1" > $TSVFILE
fi

# concatenate all tsv files into the main tsv file
cat $TSVPATH/run_*.tsv >> $TSVFILE

# remove all the tsv files
rm -rf $TSVPATH/run_*.tsv

# remove all the run scripts
rm -rf $SCRIPTPATH/run*.py

# concatenate all the log files into one main log file
cat $LOGPATH/run_*.log > $LOGFILE

# remove all the log files
rm -rf $LOGPATH/run_*.log
