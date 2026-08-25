"""A very small SM83 (Game Boy CPU) assembler.

Only exists to build the synthetic ROM in `fake_puzznic_rom.py`, which is what makes the
Puzznic Game Boy environment testable without the copyrighted cartridge. It covers the
instruction forms that ROM uses and nothing else: unsupported mnemonics raise rather than
silently assembling to the wrong bytes.

    asm = Assembler({"GRID": 0xDF00})
    asm.org(0x0150)
    asm.asm('''
        di
        ld sp, $CFFF
        jp main
    ''')
    rom = asm.link()
"""
import re

# Register encodings shared by the `ld r,r'` block and the ALU block.
R8 = {"b": 0, "c": 1, "d": 2, "e": 3, "h": 4, "l": 5, "(hl)": 6, "a": 7}
R16 = {"bc": 0, "de": 1, "hl": 2, "sp": 3}
R16_STACK = {"bc": 0, "de": 1, "hl": 2, "af": 3}
CC = {"nz": 0, "z": 1, "nc": 2, "c": 3}
ALU = {"add": 0, "adc": 1, "sub": 2, "sbc": 3, "and": 4, "xor": 5, "or": 6, "cp": 7}
CB_SHIFT = {"rlc": 0, "rrc": 1, "rl": 2, "rr": 3, "sla": 4, "sra": 5, "swap": 6, "srl": 7}
CB_BIT = {"bit": 1, "res": 2, "set": 3}
SIMPLE = {
    "nop": 0x00, "rlca": 0x07, "rrca": 0x0F, "stop": 0x10, "rla": 0x17, "rra": 0x1F,
    "daa": 0x27, "cpl": 0x2F, "scf": 0x37, "ccf": 0x3F, "halt": 0x76, "ret": 0xC9,
    "reti": 0xD9, "di": 0xF3, "ei": 0xFB,
}


class AsmError(Exception):
    pass


class Assembler:
    """Assembles into a flat ROM image. Labels may be used before they are defined."""

    def __init__(self, symbols=None, size=0x8000):
        self.image = bytearray(size)
        self.size = size
        self.pc = 0
        self.labels = dict(symbols or {})
        self.fixups = []          # (position, kind, expression, source line)
        self._scope = ""          # for `.local` labels, which belong to the last global one

    # ------------------------------------------------------------------ emitting

    def org(self, addr):
        self.pc = addr

    def byte(self, value):
        if not 0 <= value <= 0xFF:
            raise AsmError(f"byte out of range: {value}")
        self.image[self.pc] = value
        self.pc += 1

    def db(self, values):
        for value in values:
            self.byte(value)

    def dw(self, *values):
        """Little-endian words, for the pointer tables a Game Boy indexes stages with."""
        for value in values:
            if not 0 <= value <= 0xFFFF:
                raise AsmError(f"word out of range: {value}")
            self.byte(value & 0xFF)
            self.byte(value >> 8)

    def label(self, name):
        if name in self.labels:
            raise AsmError(f"duplicate label: {name}")
        self.labels[name] = self.pc

    # ------------------------------------------------------------------ operands

    def _qualify(self, name):
        """`.local` labels hang off the most recent global label."""
        return self._scope + name if name.startswith(".") else name

    def _value(self, expr, line):
        """Resolve `12`, `$0C`, `LABEL`, `LABEL+3` — or None while a label is unknown."""
        expr = expr.strip()
        match = re.fullmatch(r"([^+\-]+?)\s*(?:([+\-])\s*(.+))?", expr)
        if not match:
            raise AsmError(f"bad expression {expr!r} in {line!r}")
        head, sign, tail = match.groups()
        base = self._number(head, line)
        if base is None:
            return None
        if sign:
            offset = self._number(tail, line)
            if offset is None:
                return None
            base = base + offset if sign == "+" else base - offset
        return base

    def _number(self, token, line):
        token = token.strip()
        if re.fullmatch(r"\$[0-9a-fA-F]+", token):
            return int(token[1:], 16)
        if re.fullmatch(r"0[xX][0-9a-fA-F]+", token):
            return int(token, 16)
        if re.fullmatch(r"%[01]+", token):
            return int(token[1:], 2)
        if re.fullmatch(r"\d+", token):
            return int(token)
        name = self._qualify(token)
        if name in self.labels:
            return self.labels[name]
        if re.fullmatch(r"\.?[A-Za-z_][A-Za-z_0-9.]*", token):
            return None               # a label we have not seen yet
        raise AsmError(f"bad operand {token!r} in {line!r}")

    def _emit_n(self, expr, line):
        value = self._value(expr, line)
        if value is None:
            self.fixups.append((self.pc, "n", self._qualify(expr.strip()), line))
            self.byte(0)
        else:
            self.byte(value & 0xFF)

    def _emit_nn(self, expr, line):
        value = self._value(expr, line)
        if value is None:
            self.fixups.append((self.pc, "nn", self._qualify(expr.strip()), line))
            self.byte(0)
            self.byte(0)
        else:
            self.byte(value & 0xFF)
            self.byte((value >> 8) & 0xFF)

    def _emit_rel(self, expr, line):
        value = self._value(expr, line)
        if value is None:
            self.fixups.append((self.pc, "rel", self._qualify(expr.strip()), line))
            self.byte(0)
        else:
            self.byte(self._rel_offset(value, self.pc + 1, line))
            
    @staticmethod
    def _rel_offset(target, next_pc, line):
        delta = target - next_pc
        if not -128 <= delta <= 127:
            raise AsmError(f"jr out of range ({delta}) in {line!r} — use jp")
        return delta & 0xFF

    # ------------------------------------------------------------------ assembling

    def asm(self, source):
        for raw in source.splitlines():
            line = raw.split(";")[0].strip()
            if not line:
                continue
            while True:
                match = re.match(r"(\.?[A-Za-z_][A-Za-z_0-9.]*):\s*", line)
                if not match:
                    break
                name = match.group(1)
                if not name.startswith("."):
                    self._scope = name
                self.label(self._qualify(name))
                line = line[match.end():].strip()
            if line:
                self._instruction(line)
        return self

    def _instruction(self, line):
        parts = line.split(None, 1)
        op = parts[0].lower()
        args = [a.strip() for a in parts[1].split(",")] if len(parts) > 1 else []
        argl = [a.lower() for a in args]

        if op in SIMPLE and not args:
            return self.byte(SIMPLE[op])
        if op == "db":
            return self.db([self._value(a, line) & 0xFF for a in args])
        if op == "dw":
            for a in args:
                self._emit_nn(a, line)
            return

        handler = getattr(self, f"_op_{op}", None)
        if handler is None:
            raise AsmError(f"unsupported instruction: {line!r}")
        return handler(args, argl, line)

    # --- loads

    def _op_ld(self, args, argl, line):
        dst, src = argl[0], argl[1]
        if dst == "sp" and src == "hl":
            return self.byte(0xF9)
        if dst in R16 and src not in R8 and not src.startswith("("):
            self.byte(0x01 | (R16[dst] << 4))
            return self._emit_nn(args[1], line)
        if dst in R8 and src in R8:
            if dst == "(hl)" and src == "(hl)":
                raise AsmError("ld (hl),(hl) is halt")
            return self.byte(0x40 | (R8[dst] << 3) | R8[src])
        if dst in R8:
            if src in ("(bc)", "(de)") and dst == "a":
                return self.byte(0x0A if src == "(bc)" else 0x1A)
            if src == "(hl+)" or src == "(hli)":
                return self.byte(0x2A)
            if src == "(hl-)" or src == "(hld)":
                return self.byte(0x3A)
            if src == "(c)" and dst == "a":
                return self.byte(0xF2)
            if src.startswith("("):
                if dst != "a":
                    raise AsmError(f"only `ld a,(nn)` exists: {line!r}")
                self.byte(0xFA)
                return self._emit_nn(args[1][1:-1], line)
            self.byte(0x06 | (R8[dst] << 3))
            return self._emit_n(args[1], line)
        if src == "a":
            if dst in ("(bc)", "(de)"):
                return self.byte(0x02 if dst == "(bc)" else 0x12)
            if dst in ("(hl+)", "(hli)"):
                return self.byte(0x22)
            if dst in ("(hl-)", "(hld)"):
                return self.byte(0x32)
            if dst == "(c)":
                return self.byte(0xE2)
            if dst.startswith("("):
                self.byte(0xEA)
                return self._emit_nn(args[0][1:-1], line)
        raise AsmError(f"unsupported ld: {line!r}")

    def _op_ldh(self, args, argl, line):
        if argl[0] == "a" and argl[1].startswith("("):
            self.byte(0xF0)
            return self._emit_n(args[1][1:-1], line)
        if argl[1] == "a" and argl[0].startswith("("):
            self.byte(0xE0)
            return self._emit_n(args[0][1:-1], line)
        raise AsmError(f"unsupported ldh: {line!r}")

    # --- arithmetic

    def _op_inc(self, args, argl, line):
        if argl[0] in R16:
            return self.byte(0x03 | (R16[argl[0]] << 4))
        return self.byte(0x04 | (R8[argl[0]] << 3))

    def _op_dec(self, args, argl, line):
        if argl[0] in R16:
            return self.byte(0x0B | (R16[argl[0]] << 4))
        return self.byte(0x05 | (R8[argl[0]] << 3))

    def _alu(self, name, args, argl, line):
        # `add a, x` and `add x` both appear in the wild; so do `sub x` / `sub a, x`.
        if len(argl) == 2 and argl[0] == "a":
            args, argl = args[1:], argl[1:]
        elif len(argl) == 2 and name == "add" and argl[0] == "hl":
            return self.byte(0x09 | (R16[argl[1]] << 4))
        elif len(argl) == 2 and name == "add" and argl[0] == "sp":
            self.byte(0xE8)
            return self._emit_n(args[1], line)
        if argl[0] in R8:
            return self.byte(0x80 | (ALU[name] << 3) | R8[argl[0]])
        self.byte(0xC6 | (ALU[name] << 3))
        return self._emit_n(args[0], line)

    for _name in ALU:
        exec(f"def _op_{_name}(self, args, argl, line): return self._alu({_name!r}, args, argl, line)")
    del _name

    # --- bit operations (CB prefix)

    def _op_cb_shift(self, name, args, argl, line):
        self.byte(0xCB)
        return self.byte((CB_SHIFT[name] << 3) | R8[argl[0]])

    for _name in CB_SHIFT:
        exec(f"def _op_{_name}(self, args, argl, line): return self._op_cb_shift({_name!r}, args, argl, line)")
    del _name

    def _op_cb_bit(self, name, args, argl, line):
        index = self._value(args[0], line)
        if index is None or not 0 <= index <= 7:
            raise AsmError(f"bad bit index in {line!r}")
        self.byte(0xCB)
        return self.byte((CB_BIT[name] << 6) | (index << 3) | R8[argl[1]])

    for _name in CB_BIT:
        exec(f"def _op_{_name}(self, args, argl, line): return self._op_cb_bit({_name!r}, args, argl, line)")
    del _name

    # --- control flow

    def _op_jp(self, args, argl, line):
        if argl[0] in ("(hl)", "hl"):
            return self.byte(0xE9)
        if len(argl) == 2:
            self.byte(0xC2 | (CC[argl[0]] << 3))
            return self._emit_nn(args[1], line)
        self.byte(0xC3)
        return self._emit_nn(args[0], line)

    def _op_jr(self, args, argl, line):
        if len(argl) == 2:
            self.byte(0x20 | (CC[argl[0]] << 3))
            return self._emit_rel(args[1], line)
        self.byte(0x18)
        return self._emit_rel(args[0], line)

    def _op_call(self, args, argl, line):
        if len(argl) == 2:
            self.byte(0xC4 | (CC[argl[0]] << 3))
            return self._emit_nn(args[1], line)
        self.byte(0xCD)
        return self._emit_nn(args[0], line)

    def _op_ret(self, args, argl, line):
        return self.byte(0xC0 | (CC[argl[0]] << 3))

    def _op_push(self, args, argl, line):
        return self.byte(0xC5 | (R16_STACK[argl[0]] << 4))

    def _op_pop(self, args, argl, line):
        return self.byte(0xC1 | (R16_STACK[argl[0]] << 4))

    def _op_rst(self, args, argl, line):
        return self.byte(0xC7 | (self._value(args[0], line) & 0x38))

    # ------------------------------------------------------------------ linking

    def link(self):
        for position, kind, expr, line in self.fixups:
            value = self._value(expr, line)
            if value is None:
                raise AsmError(f"undefined label {expr!r} in {line!r}")
            if kind == "n":
                self.image[position] = value & 0xFF
            elif kind == "nn":
                self.image[position] = value & 0xFF
                self.image[position + 1] = (value >> 8) & 0xFF
            else:
                self.image[position] = self._rel_offset(value, position + 1, line)
        self.fixups = []
        return bytes(self.image)
