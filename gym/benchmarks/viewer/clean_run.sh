#!/usr/bin/env bash

bmrun_dir=$1

find $bmrun_dir \( -name *openaigym.video* -o -name log.txt -o -name events.out.tfevents.* \) -delete
