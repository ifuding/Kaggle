#! /bin/bash

in=$1
awk -F',' '
            BEGIN{min_loss=2; min_loss_row = 0;}
            {
                loss = $3
                if (loss < min_loss)
                    {
                        min_loss = loss;
                        min_loss_row = NR;
                    }
            }
            END{print min_loss, min_loss_row}
            ' ${in}
