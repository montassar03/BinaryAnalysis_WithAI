import json
import os

INPUT_JSONL  = "/home/ru94wid/projects/PalmTree/how2use/zlib_x64_blocks.jsonl"
OUTPUT_JSONL = "/home/ru94wid/projects/PalmTree/how2use/zlib_x64_instruction_embeddings.jsonl"

print("INPUT :", INPUT_JSONL)
print("OUTPUT:", OUTPUT_JSONL)

if not os.path.exists(INPUT_JSONL):
    raise FileNotFoundError(f"Input JSONL not found: {INPUT_JSONL}")

os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

total_lines = 0
parsed = 0
skipped_no_instructions = 0
written = 0
json_errors = 0

with open(INPUT_JSONL, "r") as fin, open(OUTPUT_JSONL, "w") as fout:
    for line in fin:
        total_lines += 1
        line = line.strip()
        if not line:
            continue

        try:
            rec = json.loads(line)
            parsed += 1
        except Exception as e:
            json_errors += 1
            if json_errors <= 5:
                print("JSON decode error on line", total_lines, ":", str(e))
            continue

        instructions = rec.get("instructions")
        if not instructions:
            skipped_no_instructions += 1
            continue

        # Encode per-instruction embeddings
        emb = palmtree.encode(instructions)  # (N, 128)

        # Write one line per instruction
        for ins, vec in zip(instructions, emb):
            out = {
                "project_name": rec.get("project_name"),
                "binary_name": rec.get("binary_name"),
                "target_name": rec.get("target_name"),
                "compiler": rec.get("compiler"),
                "compiler_ver": rec.get("compiler_ver"),
                "optimization_lvl": rec.get("optimization_lvl"),
                "function_name": rec.get("function_name"),
                "function_addr": rec.get("function_addr"),
                "instruction": ins,
                "embedding": vec.tolist(),
            }
            fout.write(json.dumps(out) + "\n")
            written += 1

        if parsed % 100 == 0:
            print(f"parsed={parsed} lines={total_lines} skipped_no_instructions={skipped_no_instructions} written={written}")

print("DONE")
print("total_lines:", total_lines)
print("parsed:", parsed)
print("json_errors:", json_errors)
print("skipped_no_instructions:", skipped_no_instructions)
print("written:", written)
print("output size bytes:", os.path.getsize(OUTPUT_JSONL))

#text = ["mov rbp rdi", 
 #       "mov ebx 0x1", 
  #      "mov rdx rbx", 
   #     "call memcpy", 
    #    "mov [ rcx + rbx ] 0x0", 
     #   "mov rcx rax", 
      #  "mov [ rax ] 0x2e"]

# it is better to make batches as large as possible.
#embeddings = palmtree.encode(text)
#print("usable embedding of this basicblock:", embeddings)
#print("the shape of output tensor: ", embeddings.shape)

