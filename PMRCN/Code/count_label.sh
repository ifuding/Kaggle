#! /bin/bash

in=$1
awk '{a[$1]++; tot++}END{for(x in a){print x, a[x], a[x]/tot}}' ${in}
