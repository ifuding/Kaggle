#! /bin/bash

label=$1
pred_label=$2

awk 'NR == FNR{a[NR] = $1; num[$1]++}NR != FNR{if ($1 == a[FNR]){prec[a[FNR]]++;right++}}END{print right, right / FNR; for (x in num){print x, prec[x]/num[x]}}' ${label} ${pred_label}
