import os
import numpy as np
import pandas as pd
import json
from io import StringIO
import time
import hashlib

def current_milli_time():
    return round(time.time() * 1000)

pd.set_option('display.max_columns', None)

data_dir = "/home/idomino/OU/t847/data/ember2018/"
out_dir = "/home/idomino/OU/t847/data/processed/"


def hash_string(s, m):
    return abs(hash(s)) % m

# Values
hist_size = 256
printdist_size = 96
hash_trick_dll = 128
hash_trick_import = 256
hash_trick_export = 128
hash_trick_sections = 50


coff_machines = ['AMD64', 'ARM', 'ARMNT', 'I386', 'IA64', 'MIPS16', 'MIPSFPU', 'POWERPC', 'R4000', 'SH3', 'SH4', 'THUMB']

coff_characteristics = ['AGGRESSIVE_WS_TRIM', 'BYTES_REVERSED_HI', 'BYTES_REVERSED_LO', 'CHARA_32BIT_MACHINE',
                        'DEBUG_STRIPPED', 'DLL', 'EXECUTABLE_IMAGE', 'LARGE_ADDRESS_AWARE', 'LINE_NUMS_STRIPPED',
                        'LOCAL_SYMS_STRIPPED','NET_RUN_FROM_SWAP', 'RELOCS_STRIPPED', 'REMOVABLE_RUN_FROM_SWAP',
                        'SYSTEM', 'UP_SYSTEM_ONLY']

subsystems = ['EFI_APPLICATION', 'EFI_BOOT_SERVICE_DRIVER', 'EFI_RUNTIME_DRIVER', 'NATIVE', 'POSIX_CUI', 'UNKNOWN',
              'WINDOWS_BOOT_APPLICATION', 'WINDOWS_CE_GUI', 'WINDOWS_CUI', 'WINDOWS_GUI', 'XBOX']

dll_characteristics = ['APPCONTAINER', 'DYNAMIC_BASE', 'FORCE_INTEGRITY', 'GUARD_CF', 'HIGH_ENTROPY_VA',
                       'NO_BIND', 'NO_ISOLATION', 'NO_SEH', 'NX_COMPAT', 'TERMINAL_SERVER_AWARE', 'WDM_DRIVER']

magics = ['PE32', 'PE32_PLUS']


section_props = ['ALIGN_1024BYTES', 'ALIGN_128BYTES', 'ALIGN_16BYTES', 'ALIGN_1BYTES',
                 'ALIGN_2048BYTES', 'ALIGN_256BYTES', 'ALIGN_2BYTES', 'ALIGN_32BYTES',
                 'ALIGN_4096BYTES', 'ALIGN_4BYTES', 'ALIGN_512BYTES', 'ALIGN_64BYTES',
                 'ALIGN_8192BYTES', 'ALIGN_8BYTES', 'CNT_CODE', 'CNT_INITIALIZED_DATA',
                 'CNT_UNINITIALIZED_DATA', 'GPREL', 'LNK_COMDAT', 'LNK_INFO', 'LNK_NRELOC_OVFL',
                 'LNK_OTHER', 'LNK_REMOVE', 'MEM_16BIT', 'MEM_DISCARDABLE', 'MEM_EXECUTE', 'MEM_LOCKED',
                 'MEM_NOT_CACHED', 'MEM_NOT_PAGED', 'MEM_PRELOAD', 'MEM_READ',
                 'MEM_SHARED', 'MEM_WRITE', 'TYPE_NO_PAD']

    
# Appeared
header = ''
header += 'appeared'

# Histograms
for i in range(0,hist_size):
    header += f',histogram_{i}'
for i in range(0,hist_size):
    header += f',byteentropy_{i}'

# Strings
header += ',strings_num'
header += ',strings_avlength'
for i in range(0,printdist_size):
    header += f',strings_printabledist_{i}'
header += ',strings_printables'
header += ',strings_entropy'
header += ',strings_paths'
header += ',strings_urls'
header += ',strings_registry'
header += ',strings_MZ'

# General
header += ',general_size'
header += ',general_vsize'
header += ',general_has_debug'
header += ',general_exports'
header += ',general_imports'
header += ',general_has_relocations'
header += ',general_has_resources'
header += ',general_has_signature'
header += ',general_has_tls'
header += ',general_symbols'

# Header
header += ',header_coff_timestamp'
for machine in coff_machines:
    header += f',header_coff_machine_{machine}'
for characteristic in coff_characteristics:
    header += f',header_coff_{characteristic}'
for subsys in subsystems:
    header += f',header_opt_subsystem_{subsys}'
for characteristic in dll_characteristics:
    header += f',header_opt_ddl_characteristic_{characteristic}'
for magic in magics:
    header += f',header_opt_{magic}'
header += ',header_opt_major_image_version'
header += ',header_opt_minor_image_version'
header += ',header_opt_major_linker_version'
header += ',header_opt_minor_linker_version'
header += ',header_opt_major_operating_system_version'
header += ',header_opt_minor_operating_system_version'
header += ',header_opt_major_subsystem_version'
header += ',header_opt_minor_subsystem_version'
header += ',header_opt_sizeof_code'
header += ',header_opt_sizeof_headers'
header += ',header_opt_sizeof_heap_commit'

# Sections
for i in range(0,hash_trick_sections):
    header += f',sections_h{i}_size'
    header += f',sections_h{i}_entropy'
    header += f',sections_h{i}_vsize'

for prop in section_props:
    header += f',sections_ENTRY_{prop}'
    
# Imports
for i in range(0,hash_trick_dll):
    header += f',imports_dll_h{i}_imported'
for i in range(0,hash_trick_import):
    header += f',imports_fun_h{i}_imported'

# Exports
for i in range(0,hash_trick_export):
    header += f',exports_h{i}'
    
# Control
header += ',label'
header += ',avclass'
    
    
def line_to_row(line):
    # Appeared
    row = ''
    data = json.loads(line)
    row += data['appeared'] + ','
    
    # Histograms
    for i in range(0,hist_size):
        row += str(data['histogram'][i]) + ','
    for i in range(0,hist_size):
        row += str(data['byteentropy'][i]) + ','
    
    # Strings
    row += str(data['strings']['numstrings']) + ','
    row += str(data['strings']['avlength']) + ','
    for i in range(0,printdist_size):
        row += str(data['strings']['printabledist'][i]) + ','
    row += str(data['strings']['printables']) + ','
    row += str(data['strings']['entropy']) + ','
    row += str(data['strings']['paths']) + ','
    row += str(data['strings']['urls']) + ','
    row += str(data['strings']['registry']) + ','
    row += str(data['strings']['MZ']) + ','
    
    # General
    row += str(data['general']['size']) + ','
    row += str(data['general']['vsize']) + ','
    row += str(data['general']['has_debug']) + ','
    row += str(data['general']['exports']) + ','
    row += str(data['general']['imports']) + ','
    row += str(data['general']['has_relocations']) + ','
    row += str(data['general']['has_resources']) + ','
    row += str(data['general']['has_signature']) + ','
    row += str(data['general']['has_tls']) + ','
    row += str(data['general']['symbols']) + ','
    
    # Header
    row += str(data['header']['coff']['timestamp']) + ','
    for machine in coff_machines:
        row += ('1' if data['header']['coff']['machine'] == machine else '0') + ','
    for characteristic in coff_characteristics:
        row += ('1' if characteristic in data['header']['coff']['characteristics'] else '0') + ','
    for subsys in subsystems:
        row += ('1' if data['header']['optional']['subsystem'] == subsys else '0') + ','
    for characteristic in dll_characteristics:
        row += ('1' if characteristic in data['header']['optional']['dll_characteristics'] else '0') + ','
    for magic in magics:
        row += ('1' if magic == data['header']['optional']['magic'] else '0') + ','
    row += str(data['header']['optional']['major_image_version']) + ','
    row += str(data['header']['optional']['minor_image_version']) + ','
    row += str(data['header']['optional']['major_linker_version']) + ','
    row += str(data['header']['optional']['minor_linker_version']) + ','
    row += str(data['header']['optional']['major_operating_system_version']) + ','
    row += str(data['header']['optional']['minor_operating_system_version']) + ','
    row += str(data['header']['optional']['major_subsystem_version']) + ','
    row += str(data['header']['optional']['minor_subsystem_version']) + ','
    row += str(data['header']['optional']['sizeof_code']) + ','
    row += str(data['header']['optional']['sizeof_headers']) + ','
    row += str(data['header']['optional']['sizeof_heap_commit']) + ','
    
    # Sections
    entry_section = None
    section_dict = {}
    for i in data['section']['sections']:
        if i['name'] == data['section']['entry']:
            entry_section = i
        section_dict[hash_string(i['name'],hash_trick_sections)] = i
        
    for i in range(0,hash_trick_sections):
        section_data = section_dict.get(i)
        if section_data:
            row += str(section_data['size']) + ','
            row += str(section_data['entropy']) + ','
            row += str(section_data['vsize']) + ','
        else:
            row += '0,'
            row += '0,'
            row += '0,'
    
    if entry_section:
        entry_props = entry_section['props']
        for prop in section_props:
            row += ('1' if prop in entry_props else '0') + ','
    else:
        for prop in section_props:
            row += '0,'
    
    # Imports
    for i in range(0,hash_trick_dll):
        hashed_dlls = [hash_string(x, hash_trick_dll) for x in data['imports']]
        row += ('1' if i in hashed_dlls else '0') + ','
    for i in range(0,hash_trick_import):
        imported = False
        for key in data['imports']:
            hashed_funcs = [hash_string(f'{key}:{x}', hash_trick_import) for x in data['imports'][key]]
            if i in hashed_funcs:
                imported = True
        row += ('1' if imported else '0') + ','
    
    # Exports
    for i in range(0,hash_trick_export):
        hashed_exports = [hash_string(x, hash_trick_export) for x in data['exports']]
        row += ('1' if i in hashed_exports else '0') + ','
    
    # Labels
    row += str(data['label']) + ','
    row += str(data['avclass']) if data['avclass'] else '-'
    
    return row

test_data = header
with open(data_dir + 'train_features_0.jsonl', 'r') as f:
    for i in range(0,5):
        line = f.readline()
        test_data += '\n'
        test_data += line_to_row(line)

df = pd.read_csv(StringIO(test_data))
print(df.dtypes)
df.columns


datafiles = ['train_features_0.jsonl', 'train_features_1.jsonl',
'train_features_2.jsonl','train_features_3.jsonl',
'train_features_4.jsonl','train_features_5.jsonl','test_features.jsonl']

def save_buffer(buffer, fragment):
    df = pd.read_csv(StringIO(header + buffer))
    df.to_pickle(f'{out_dir}data{fragment}.pkl', compression='zip')
    return ('', fragment+1)

t0 = current_milli_time()
buffer = ''
fragment = 0
chunksize = 50000
i = 0
for datafile in datafiles:
    print('Datafile:', datafile)
    with open(data_dir + datafile, 'r') as infile:
        while True:
            line = infile.readline()
            if not line:
                break
            row = line_to_row(line)
            buffer += '\n'
            buffer += row
            i += 1
            if i % chunksize == 0:
                t1 = current_milli_time()
                print(f'[{int((t1-t0)/1000)}]','Iteration:', i)
                buffer, fragment = save_buffer(buffer, fragment)
if buffer:
    save_buffer(buffer, fragment)
