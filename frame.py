import wx
import wx.lib.statbmp as SB
from pubsub import pub
from PIL import Image

PhotoWeight = 512
PhotoHeight = 384


class Frame(wx.Frame):
    def __init__(self, parent, title):
        super(Frame, self).__init__(parent, title=title)
        self.panel = wx.Panel(self)
        st = wx.StaticText(self.panel, label="Sprawdź do jakiego pojemnika możesz\nwrzucić odpad",
                           style=wx.ALIGN_CENTER)

        font = st.GetFont()

        font.PointSize += 5
        font = font.Bold()
        st.SetFont(font)

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.sizer.Add(st, wx.SizerFlags().Border(wx.LEFT, 90))
        self.panel.SetSizer(self.sizer)

        button = wx.Button(self.panel, wx.ID_ANY, 'Sprawdź gdzie wyrzucić', (170, 550))

        button.SetBackgroundColour((255, 255, 255, 1))
        button.SetWindowStyleFlag(wx.SIMPLE_BORDER)
        button.Bind(wx.EVT_BUTTON, onButton)

        bitmap = wx.Bitmap("images/recycling.jpg")
        img = bitmap.ConvertToImage()
        img = img.Scale(512, 384, wx.IMAGE_QUALITY_HIGH)

        self.image_ctrl = SB.GenStaticBitmap(self.panel, wx.ID_ANY, wx.Bitmap(img))

        image_drop = DropImage(self)

        self.image_ctrl.SetDropTarget(image_drop)
        self.sizer.Add(self.image_ctrl, 0, wx.ALL, 5)
        self.panel.SetSizer(self.sizer)
        self.sizer.Add(button, wx.SizerFlags().Border(wx.LEFT, 200))
        self.panel.Refresh()
        self.panel.SetSizer(self.sizer)
        self.sizer.Fit(self)

        pub.subscribe(self.update_image_on_drag, 'imgDragged')

    def update_image_on_drag(self, filepath):
        self.on_view(filepath=filepath)

    def on_view(self, filepath):
        img = wx.Image(filepath, wx.BITMAP_TYPE_ANY)

        w = img.GetWidth()
        h = img.GetHeight()

        img = img.Scale(w, h)
        self.image_ctrl.SetBitmap(wx.Bitmap(img))
        self.panel.Refresh()


class DropImage(wx.FileDropTarget):

    def __init__(self, widget):
        wx.FileDropTarget.__init__(self)
        self.widget = widget

    def OnDropFiles(self, x, y, filenames):
        image = Image.open(filenames[0])
        image.thumbnail((PhotoWeight, PhotoHeight))
        image.save('images/img_to_check.png')
        pub.sendMessage('imgDragged', filepath='images/img_to_check.png')
        return True


class MyApp(wx.App):
    def __init__(self, redirect=False, filename=None, useBestVisual=False, clearSigInt=True):
        super().__init__(redirect, filename, useBestVisual, clearSigInt)
        self.frame = None

    def OnInit(self):
        self.frame = Frame(parent=None, title="Trash sorter")
        self.frame.Show()

        return True


def onButton(event):
    print("klik")


app = MyApp()
app.MainLoop()
