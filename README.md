You will need to install the following dependencies:
```pip install numpy casadi matplotlib scipy proxsuite```

If you have problems with proxsuite, you can try a different QP solver, such as
```pip install osqp```
but you also need to change ```opt.solver('proxqp')``` to ```opt.solver('osqp')``` in a couple of files
