from SpiNNiCub.connection_matrix import iCub_main



# filepath name
CFG_FILE_NAME = "spynnaker.cfg"

# the cfg params basic needs
BASIC_DATA = (
        "[Buffers]\n\n" + "use_auto_pause_and_resume = True\n\n" +
        "[Simulation]\n\n" + "incoming_spike_buffer_size = 512\n\n" +
        "[Reports]\n\n" + "extract_iobuf = False\n\n" +
        "extract_iobuf_during_run = False\n\n" +
        "clear_iobuf_during_run = False\n\n" +
        "[Machine] \n\ntimeScaleFactor = ")

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




if __name__ == '__main__':
    timescale = 1
    set_config_file(two=True,
                    twone=True,
                    three=False,
                    four=False,
                    five=False,
                    time_scale_factor=timescale)
    iCub_main()
