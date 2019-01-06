# Fit basic stuff for experimental master thesis

Two Python implementations of fitting a sum of exponentials to numerical data.
I haven't tested both carefully, but method 2 seems to work better:


* Based on Kaufmann (2003) "Fitting a Sum of Exponentials to Numerical Data".
* Based on scipy.optimize differential evolution

**Note: The kaufmann implementation finds the best fit assuming the exponential does not decay to zero, whereas the diffevol implementation does assume the exponential decays to zero.**

They only deal with negative decay sums :/

```
In [6]: run main
   Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality
       0              1         3.6198e+10                                    2.85e+11
       1              2         1.6867e+08      3.60e+10       5.59e+01       1.50e+08
       2              3         9.4298e+07      7.44e+07       9.80e+01       2.71e+09
       3              9         8.5141e+07      9.16e+06       1.69e-01       1.13e+08
       4             10         7.9132e+07      6.01e+06       4.84e-02       8.17e+07
       5             11         7.1444e+07      7.69e+06       8.71e-02       6.71e+07
       6             12         6.6601e+07      4.84e+06       6.28e-02       3.65e+07
       7             13         6.4612e+07      1.99e+06       4.94e-02       3.58e+07
       8             14         5.9279e+07      5.33e+06       1.45e-01       3.41e+07

```
