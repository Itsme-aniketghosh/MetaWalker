"""Work around a nevergrad / numpy 2.x crash (metamodel.py calls float() on a 1-element array). Idempotent;
cluster/setup.sbatch runs it after installing nevergrad."""
import nevergrad.optimization.metamodel as mm
p = mm.__file__
s = open(p).read()
old = "return float(model.predict(trans(np.asarray(x, dtype=X[0].dtype).flatten()[None, :])))"
new = "return float(np.ravel(model.predict(trans(np.asarray(x, dtype=X[0].dtype).flatten()[None, :])))[0])"
if new in s:
    print('already patched', p)
else:
    assert s.count(old) == 1, 'expected line not found; nevergrad version changed?'
    open(p, 'w').write(s.replace(old, new)); print('patched', p)
