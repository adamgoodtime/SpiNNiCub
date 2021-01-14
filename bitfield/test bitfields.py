from SpiNNiCub.connection_matrix import iCub_main



# filepath name
CFG_FILE_NAME = "spynnaker.cfg"

# the cfg params basic needs
BASIC_DATA = (
        "[Buffers]\n\n" + "use_auto_pause_and_resume = True\n\n" +
        "[Simulation]\n\n" + "incoming_spike_buffer_size = 1024\n\n" +
        "[Reports]\n\n" + "extract_iobuf = False\n\n" +
        "extract_iobuf_during_run = False\n\n" +
        "clear_iobuf_during_run = False\n\n" +
        "write_compressor_iobuf = True\n\n"
        "[Machine] \n\ntimeScaleFactor = ")
'''
[Mapping]
router_table_compression_with_bit_field_use_time_cutoff = True
router_table_compression_with_bit_field_iteration_time = 1000
router_table_compression_with_bit_field_pre_alloced_sdram = 10000
router_table_compression_with_bit_field_acceptance_threshold = 0
'''

def set_config_file(two, twone, three, four, five, time_scale_factor=1):
    """ sets the spynnaker.cfg depending on setup

    :return:
    """
    output = open(CFG_FILE_NAME, "w")

    if (two and three) or (two and four) or (three and four):
        print("multiple loading algorithms, probably not ok")
        Exception
    if (twone and five):
        print("multiple machine algorithms, probably not ok")
        Exception

    output.write(BASIC_DATA + "{}\n\n".format(time_scale_factor))
    output.write("[Mapping] \n\n")
    # output.write('router_table_compression_with_bit_field_iteration_time = 100000\n')  # broken?
    if two:
        output.write("loading_algorithms = MundyOnChipRouterCompression\n")
    if twone:
        output.write("machine_graph_to_machine_algorithms = EdgeToNKeysMapper,OneToOnePlacer,NerRoute,BasicTagAllocator,ProcessPartitionConstraints,MallocBasedRoutingInfoAllocator,BasicRoutingTableGenerator,RouterCollisionPotentialReport\n")
    if three:
        output.write("loading_algorithms = SpynnakerMachineBitFieldUnorderedRouterCompressor,BitFieldCompressorReport\n")
    if four:
        output.write("loading_algorithms = SpynnakerMachineBitFieldPairRouterCompressor,BitFieldCompressorReport\n")
    if five:
        output.write("machine_graph_to_machine_algorithms = EdgeToNKeysMapper,SpreaderPlacer,NerRoute,BasicTagAllocator,ProcessPartitionConstraints,MallocBasedRoutingInfoAllocator,BasicRoutingTableGenerator,RouterCollisionPotentialReport\n")
    output.flush()
    output.close()

def set_config(config, timescale):
    if config == 1:
        config = [1, 0, 0, 0, 0, timescale]
    elif config == 2:
        config = [1, 1, 0, 0, 0, timescale]
    elif config == 3:
        config = [0, 1, 1, 0, 0, timescale]
    elif config == 4:
        config = [0, 1, 0, 1, 0, timescale]
    elif config == 5:
        config = [0, 0, 1, 0, 1, timescale]
    elif config == 6:
        config = [0, 0, 0, 1, 1, timescale]
    set_config_file(*config)


if __name__ == '__main__':
    timescale = 1
    simulate = 'average'
    noise_level = 0.4
    npc = 64
    # average_rate = 220
    # average_rate = 210
    # average_rate = 200
    # average_rate = 190
    # average_rate = 180
    average_rate = 170
    config = 5
    set_config(config, timescale)
    # set_config_file(two=False,
    #                 twone=False,
    #                 three=False,
    #                 four=True,
    #                 five=True,
    #                 time_scale_factor=timescale)
    print(timescale, simulate, noise_level, config, average_rate, npc)
    iCub_main(simulate, noise_level, False, average_rate=average_rate, npc=npc)
    print("done", timescale, simulate, noise_level, config, average_rate, npc)
    print(timescale, simulate, noise_level, config, average_rate, npc)
    print("done", timescale, simulate, noise_level, config, npc)
