"""The RASP model as a Haiku/JAX model."""


from tracr import compiler
from tracr.rasp import rasp
from tracr.compiler.assemble import AssembledTransformerModel
from tracr.compiler.lib import make_frac_prevs


haiku_model: AssembledTransformerModel = (
    compiler.compiling.compile_rasp_to_model(
        make_frac_prevs(rasp.tokens == "x"),
        vocab={"w", "x", "y", "z"},
        max_seq_len=10,
        compiler_bos="BOS",
    )
)
