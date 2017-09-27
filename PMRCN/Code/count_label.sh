#! /bin/bash

in=$1
awk '{a[$2]++; tot++}END{for(x in a){print x, a[x], a[x]/tot}}' ${in}
