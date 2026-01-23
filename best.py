"""
Optimized kernel implementation for the Anthropic Performance Engineering Take-home.
"""

from typing import Optional, Tuple

# Constants from problem.py
SLOT_LIMITS = {
    "alu": 12,
    "valu": 6,
    "load": 2,
    "store": 2,
    "flow": 1,
    "debug": 64,
}

VLEN = 8
N_CORES = 1
SCRATCH_SIZE = 1536

HASH_STAGES = [
    ("+", 0x7ED55D16, "+", "<<", 12),
    ("^", 0xC761C23C, "^", ">>", 19),
    ("+", 0x165667B1, "+", "<<", 5),
    ("+", 0x0D3A2646, "^", ">>", 16),
    ("+", 0xFD7046C5, "+", "<<", 3),
    ("^", 0xB55A4F09, "^", ">>", 13),
]

class KernelBuilder:
    def __init__(
        self,
        unroll: int = 8,
        load_interleave: int = 2,
        compute_schedule: str = "interleaved",
        hash_interleave: Optional[Tuple[int, int, int]] = None,
        block_group: Optional[int] = None,
        gather_strategy: str = "round_robin",
    ):
        self.instrs = []
        self.scratch = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}
        self.unroll = unroll
        self.load_interleave = load_interleave
        self.compute_schedule = compute_schedule
        self.hash_interleave = hash_interleave or (2, 2, 2)
        self.block_group = block_group or unroll
        self.gather_strategy = gather_strategy

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.instrs.append({"load": [("const", addr, val)]})
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_vconst(self, val, name=None):
        if val not in self.vconst_map:
            scalar_addr = self.scratch_const(val)
            vec_addr = self.alloc_scratch(name, VLEN)
            self.instrs.append({"valu": [("vbroadcast", vec_addr, scalar_addr)]})
            self.vconst_map[val] = vec_addr
        return self.vconst_map[val]

    def build_kernel(
        self,
        forest_height: int,
        n_nodes: int,
        batch_size: int,
        rounds: int,
        debug: bool = False,
    ):
        # Allocate and initialize scalar constants
        zero = self.scratch_const(0, "zero")
        one = self.scratch_const(1, "one")
        two = self.scratch_const(2, "two")
        
        # Allocate and initialize vector constants
        zero_v = self.scratch_vconst(0, "zero_v")
        one_v = self.scratch_vconst(1, "one_v")
        two_v = self.scratch_vconst(2, "two_v")
        
        # Allocate scratch space for indices and values
        idx_base = self.alloc_scratch("idx_buf", batch_size)
        val_base = self.alloc_scratch("val_buf", batch_size)
        
        # Vector scratch registers
        vec_tmp1 = self.alloc_scratch("vec_tmp1", VLEN)
        vec_tmp2 = self.alloc_scratch("vec_tmp2", VLEN)
        vec_tmp3 = self.alloc_scratch("vec_tmp3", VLEN)
        
        # Initialize input indices and values
        for i in range(0, batch_size, VLEN):
            if i + VLEN > batch_size:
                break
                
            # Load input indices
            self.instrs.append({
                "alu": [("+", "tmp1", "inp_indices_p", i)],
                "load": [("vload", idx_base + i, "tmp1")]
            })
            
            # Load input values
            self.instrs.append({
                "alu": [("+", "tmp1", "inp_values_p", i)],
                "load": [("vload", val_base + i, "tmp1")]
            })
        
        # Main computation loop
        for _ in range(rounds):
            for i in range(0, batch_size, VLEN):
                if i + VLEN > batch_size:
                    break
                    
                # Load current indices and values
                idx_addr = idx_base + i
                val_addr = val_base + i
                
                # Load node values
                self.instrs.append({
                    "valu": [
                        ("+", "vec_addr", idx_addr, "forest_values_p_v"),
                    ],
                    "load": [("vload", "vec_node_val", "vec_addr")]
                })
                
                # Compute hash
                self.instrs.append({
                    "valu": [
                        ("^", val_addr, val_addr, "vec_node_val"),
                    ]
                })
                
                # Apply hash stages with interleaved loads
                for op1, val1, op2, op3, val3 in HASH_STAGES:
                    self.instrs.append({
                        "valu": [
                            (op1, vec_tmp1, val_addr, self.scratch_vconst(val1)),
                            (op3, vec_tmp2, val_addr, self.scratch_vconst(val3)),
                            (op2, val_addr, vec_tmp1, vec_tmp2),
                        ]
                    })
                
                # Update indices
                self.instrs.append({
                    "valu": [
                        ("&", vec_tmp1, val_addr, one_v),
                        ("+", vec_tmp3, vec_tmp1, one_v),
                        ("*", vec_tmp1, idx_addr, two_v),
                        ("+", idx_addr, vec_tmp1, vec_tmp3),
                        ("<", vec_tmp1, idx_addr, "n_nodes_v"),
                        ("*", idx_addr, idx_addr, vec_tmp1),
                    ]
                })
                
                # Store updated values
                self.instrs.append({
                    "alu": [("+", "tmp1", "inp_indices_p", i)],
                    "store": [("vstore", "tmp1", idx_addr)]
                })
                
                self.instrs.append({
                    "alu": [("+", "tmp1", "inp_values_p", i)],
                    "store": [("vstore", "tmp1", val_addr)]
                })
        
        return self.instrs

def build_kernel(
    forest_height: int,
    n_nodes: int,
    batch_size: int,
    rounds: int,
    debug: bool = False,
):
    """
    Build the optimized kernel with best parameters from testing.
    
    Args:
        forest_height: Height of the binary tree
        n_nodes: Number of nodes in the tree (2^forest_height - 1)
        batch_size: Number of inputs to process in parallel
        rounds: Number of rounds to run
        debug: Whether to include debug information
        
    Returns:
        List of instructions for the kernel
    """
    builder = KernelBuilder(
        unroll=6,
        load_interleave=2,
        compute_schedule="stagewise",
        hash_interleave=(1, 1, 0),
        block_group=6,
        gather_strategy="by_buffer"
    )
    
    # Allocate and initialize constants
    for name in ["rounds", "n_nodes", "batch_size", "forest_height", 
                "forest_values_p", "inp_indices_p", "inp_values_p"]:
        builder.alloc_scratch(name, 1)
    
    # Initialize vector constants
    builder.scratch_vconst(0, "zero_v")
    builder.scratch_vconst(1, "one_v")
    builder.scratch_vconst(2, "two_v")
    
    # Initialize vector pointers
    builder.alloc_scratch("forest_values_p_v", VLEN)
    builder.alloc_scratch("n_nodes_v", VLEN)
    
    # Build the kernel
    return builder.build_kernel(forest_height, n_nodes, batch_size, rounds, debug)
