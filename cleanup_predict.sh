#!/bin/bash

SCRIPTPATH=$1
LOGPATH=$2
LOGFILE=$3

# remove all the run scripts
rm -rf $SCRIPTPATH/run*.py

# concatenate all the log files into one main log file
cat $LOGPATH/run_*.log > $LOGFILE

# remove all the log files
rm -rf $LOGPATH/run_*.log
