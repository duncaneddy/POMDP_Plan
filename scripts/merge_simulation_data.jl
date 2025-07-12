#!/usr/bin/env julia

using JSON
using ArgParse
using Dates

function parse_commandline()
    s = ArgParseSettings(
        description = "Merge simulation_data arrays from multiple JSON files",
        add_help = true,
    )
    
    @add_arg_table! s begin
        "input_files"
            help = "Input JSON files containing simulation_data arrays"
            nargs = '+'
            required = true
        "--output", "-o"
            help = "Output JSON filename"
            arg_type = String
            required = true
        "--verbose", "-v"
            help = "Enable verbose output"
            action = :store_true
        "--preserve-metadata"
            help = "Preserve other fields from the first input file"
            action = :store_true
    end
    
    return parse_args(s)
end

function merge_simulation_data(input_files::Vector{String}, output_file::String; 
                             verbose::Bool=false, preserve_metadata::Bool=false)
    
    merged_data = []
    metadata = Dict()
    total_files = length(input_files)
    
    if verbose
        println("Merging simulation_data from $total_files files...")
    end
    
    for (i, file_path) in enumerate(input_files)
        if verbose
            println("Processing file $i/$total_files: $file_path")
        end
        
        # Check if file exists
        if !isfile(file_path)
            error("File not found: $file_path")
        end
        
        try
            # Load JSON file
            data = JSON.parsefile(file_path)
            
            # Check if simulation_data field exists
            if !haskey(data, "simulation_data")
                @warn "File $file_path does not contain 'simulation_data' field, skipping..."
                continue
            end
            
            sim_data = data["simulation_data"]
            
            # Verify it's an array
            if !isa(sim_data, Vector)
                @warn "simulation_data in $file_path is not an array, skipping..."
                continue
            end
            
            # Append to merged data
            append!(merged_data, sim_data)
            
            if verbose
                println("  Added $(length(sim_data)) simulation entries")
            end
            
            # Preserve metadata from first file if requested
            if i == 1 && preserve_metadata
                metadata = deepcopy(data)
                delete!(metadata, "simulation_data")  # Remove sim data, we'll add merged version
            end
            
        catch e
            @warn "Error processing file $file_path: $e"
            continue
        end
    end
    
    if isempty(merged_data)
        error("No simulation data found in any input files")
    end
    
    if verbose
        println("Total merged simulation entries: $(length(merged_data))")
    end
    
    # Create output structure
    output_data = Dict{String, Any}()
    
    if preserve_metadata && !isempty(metadata)
        # Copy metadata fields
        for (key, value) in metadata
            output_data[key] = value
        end
    end
    
    # Add simulation data
    output_data["simulation_data"] = merged_data
    
    # Add merge metadata
    output_data["merge_info"] = Dict{String, Any}(
        "source_files" => input_files,
        "total_entries" => length(merged_data),
        "merge_timestamp" => string(Dates.now())
    )
    
    # Write merged data to output file
    try
        open(output_file, "w") do f
            JSON.print(f, output_data)
        end
        
        if verbose
            println("Successfully wrote merged data to: $output_file")
        end
        
    catch e
        error("Failed to write output file $output_file: $e")
    end
    
    return length(merged_data)
end

function main()
    args = parse_commandline()
    
    try
        # Ensure input_files is a Vector{String}
        input_files = String.(args["input_files"])
        output_file = String(args["output"])
        verbose = Bool(args["verbose"])
        preserve_metadata = Bool(args["preserve-metadata"])
        
        num_entries = merge_simulation_data(
            input_files,
            output_file;
            verbose=verbose,
            preserve_metadata=preserve_metadata
        )
        
        println("✓ Successfully merged $(num_entries) simulation entries")
        return 0
        
    catch e
        println("✗ Error: $e")
        return 1
    end
end

# Run main function if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end