import sys, ismrmrd
d = ismrmrd.Dataset(sys.argv[1], "dataset", create_if_needed=False)
h = ismrmrd.xsd.CreateFromDocument(d.read_xml_header()); e = h.encoding[0]
a = d.read_acquisition(0)
print("sequence      :", getattr(h.measurementInformation, "protocolName", "?"))
print("encoded matrix: %sx%sx%s" % (e.encodedSpace.matrixSize.x, e.encodedSpace.matrixSize.y, e.encodedSpace.matrixSize.z))
print("encoded FOV mm: %sx%sx%s" % (e.encodedSpace.fieldOfView_mm.x, e.encodedSpace.fieldOfView_mm.y, e.encodedSpace.fieldOfView_mm.z))
print("acquisitions  :", d.number_of_acquisitions())
print("coils, samples:", a.data.shape)
print("embedded traj :", None if a.traj is None or a.traj.size == 0 else a.traj.shape)
d.close()
